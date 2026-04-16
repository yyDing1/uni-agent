import numpy as np
import orjson as json
import faiss
import os
import glob
from tqdm import tqdm
import multiprocessing as mp
import pyarrow.parquet as pq
import sys
from typing import List, Tuple, Optional

# --- 配置常量 ---
LOCAL_DATA_DIR = "/mnt/hdfs/wdl/wiki24-raw/data/en" 
VECTOR_DIMENSION = 1024
INDEX_PATH = "/mnt/hdfs/wdl/wiki24/wiki24_faiss.index"
TEXT_DATA_PATH = "/mnt/hdfs/wdl/wiki24/wiki24_data.jsonl" 

# FAISS IVF 参数
NLIST = 4096                # 聚类中心的数量 (Inverted List count)
TRAINING_SAMPLES = 2000000  # 用于训练聚类中心的向量数量
FAISS_METRIC = faiss.METRIC_L2 # 使用 L2 距离

# 并行化与性能参数
NUM_PROCESSES = mp.cpu_count() 
if NUM_PROCESSES > 64:
    NUM_PROCESSES = 96 

# --- 辅助函数：子进程任务 ---
def process_parquet_file(file_path: str) -> Tuple[Optional[np.ndarray], Optional[List[bytes]]]:
    """
    子进程函数：读取单个 Parquet 文件，提取向量和文档数据。
    """
    try:
        table = pq.read_table(file_path)
        data_df = table.to_pandas()
        
        embeddings = np.stack(data_df["embedding"].to_numpy())
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        text_batch_lines = []
        # 使用 iterrows 逐行处理文档元数据
        for _, row in data_df.iterrows():
            doc_data = {
                "id": str(row["id"]),
                "url": row["url"],
                "title": row["title"],
                "text": row["text"],
            }
            json_bytes = json.dumps(doc_data) + b'\n'
            text_batch_lines.append(json_bytes)
            
        return embeddings, text_batch_lines

    except Exception as e:
        print(f"Error processing file {file_path}: {e}", file=sys.stderr)
        return None, None


# --- 主程序：IVF 索引构建与多进程管理 ---
def build_faiss_index_ivf_parallel():
    
    # 1. 收集所有 Parquet 文件
    parquet_files = sorted(glob.glob(os.path.join(LOCAL_DATA_DIR, "*.parquet")))
    if not parquet_files:
        print(f"错误：未在目录 '{LOCAL_DATA_DIR}' 中找到任何 Parquet 文件。请检查路径。")
        return

    print(f"Found {len(parquet_files)} Parquet files to process.")
    
    # 2. Stage 1: 并行 I/O 和数据预处理 (Blocking)
    all_results = []
    vectors_processed = 0
    try:
        # 使用上下文管理器管理进程池
        with mp.Pool(processes=NUM_PROCESSES) as pool:
            print(f"Stage 1: Starting {NUM_PROCESSES} workers for Parallel Data Collection. Collecting to RAM...")

            results_iterator = pool.imap_unordered(process_parquet_file, parquet_files)

            pbar = tqdm(
                results_iterator, 
                total=len(parquet_files), 
                desc="Collecting All Data Chunks to RAM", 
                unit='file'
            )
            
            for embeddings, text_batch_lines in pbar:
                if embeddings is not None:
                    all_results.append((embeddings, text_batch_lines))
                    vectors_processed += len(embeddings)
                    pbar.set_postfix({"Total Docs": f"{vectors_processed:,}"})

            pbar.close()

        print(f"Stage 1 Complete. Total collected documents: {vectors_processed:,}. Now proceeding to training.")

    except KeyboardInterrupt:
        print("\n用户中断。数据收集阶段中止。")
        return
    except Exception as e:
        print(f"\nStage 1 (Data Collection) 发生错误: {e}")
        return
        
    if vectors_processed == 0:
        print("未收集到任何向量，索引构建中止。")
        return

    # 3. Stage 2: 索引训练 (并行 K-Means)
    print(f"Stage 2: Training Index (NLIST={NLIST}). Using {TRAINING_SAMPLES:,} samples...")
    
    # 提取训练数据
    training_vectors_list = []
    current_count = 0
    for embeddings, _ in all_results:
        if current_count < TRAINING_SAMPLES:
            take = min(TRAINING_SAMPLES - current_count, len(embeddings))
            training_vectors_list.append(embeddings[:take])
            current_count += take
            
    training_matrix = np.concatenate(training_vectors_list, axis=0)
    
    # 初始化 IVF 索引
    quantizer = faiss.IndexFlatL2(VECTOR_DIMENSION)
    final_index = faiss.IndexIVFFlat(quantizer, VECTOR_DIMENSION, NLIST, FAISS_METRIC)
    
    # K-Means 训练在这里临时使用所有的 CPU 核加速
    faiss.omp_set_num_threads(NUM_PROCESSES) 
    
    final_index.train(training_matrix)
    
    # 训练结束后，将 FAISS 线程数设回 1 (将核心留给操作系统和添加操作)
    faiss.omp_set_num_threads(1)
    print("Index Training Complete.")

    # 4. Stage 3: 添加向量和写入 JSONL (串行但快速)
    print("⚡ Stage 3: Adding Vectors and Writing JSONL.")
    current_idx = 0
    try:
        with open(TEXT_DATA_PATH, 'wb') as f_out:
            pbar = tqdm(all_results, desc='Adding to Index', unit='batch')
            for embeddings, text_batch_lines in pbar:
                
                # 4.1. 写入 JSONL 文件 (串行安全写入)
                f_out.writelines(text_batch_lines)
                
                # 4.2. 添加到 FAISS 索引 (串行但快速)
                final_index.add(embeddings)
                current_idx += len(embeddings)

                pbar.set_postfix({"Total Docs": f"{current_idx:,}"})

            pbar.close()

    except Exception as e:
        print(f"\nStage 3 (Add/Write) 发生错误: {e}")
        print(f"已处理向量数量: {final_index.ntotal}")
    
    # 5. 保存索引
    if final_index.ntotal > 0:
        print(f"\nFinalizing and saving FAISS index to {INDEX_PATH}...")
        faiss.write_index(final_index, INDEX_PATH)
        print(f"Index successfully saved with {final_index.ntotal:,} vectors.")
    else:
        print("索引中没有向量，跳过保存。")


if __name__ == "__main__":
    # 强制设置启动方法为 'spawn' (在 Linux/macOS 上更安全)
    if os.name != 'nt':
         mp.set_start_method('spawn', force=True)
         
    build_faiss_index_ivf_parallel()