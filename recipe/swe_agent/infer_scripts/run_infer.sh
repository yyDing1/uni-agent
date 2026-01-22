datetime=$(date +%Y%m%d_%H%M%S)
log_dir=/tmp/logs/run_${datetime}
hdfs_dir=/mnt/hdfs/yyding/swe-run/${datetime}
echo "log_dir: ${log_dir}"
# ray job submit --working-dir . -- python recipe/swe_agent/infer_only_scripts/copy_model.py
ray job submit --no-wait --working-dir . -- python recipe/swe_agent/infer_only_scripts/run_infer.py --log-dir ${log_dir}
ray job submit --no-wait --working-dir . -- python recipe/swe_agent/infer_only_scripts/sync_logs.py \
    --src ${log_dir} --dst ${hdfs_dir}
