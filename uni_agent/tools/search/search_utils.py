# modified from verl/tools/utils/search_r1_like_utils.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import threading
import time
import traceback
import uuid
from typing import Any, Optional, Callable, Dict, List
from datetime import datetime

import requests
import os

DEFAULT_TIMEOUT = 30  # Default request timeout
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def call_api(
    url: str,
    payload: dict,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    initial_retry_delay: int = INITIAL_RETRY_DELAY,
    log_prefix: str = "",
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    last_error = None

    for attempt in range(max_retries):
        try:
            logger.info(f"{log_prefix}Attempt {attempt + 1}/{max_retries}: Calling API at {url}")
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt {attempt + 1}/{max_retries}"
                logger.warning(last_error)
                if attempt < max_retries - 1:
                    delay = initial_retry_delay * (attempt + 1)
                    logger.warning(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(f"{log_prefix}API call successful on attempt {attempt + 1}")
            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < max_retries - 1:
                delay = initial_retry_delay * (attempt + 1)
                logger.warning(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < max_retries - 1:
                delay = initial_retry_delay * (attempt + 1)
                logger.warning(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on other request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}API call failed. Last error: {last_error}")
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"

def call_search_api(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    trajectory: dict[str, Any] = None
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    request_id = trajectory["id"] if trajectory and "id" in trajectory else str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "
    payload = {"queries": query_list, "topk": topk, "return_scores": return_scores}
    return call_api(retrieval_service_url, payload, timeout=timeout, log_prefix=log_prefix)

def call_serper_api_direct(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    timeout: int = DEFAULT_TIMEOUT,
    serper_api_key: str = None,
    trajectory: dict[str, Any] = None
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    request_id = trajectory["id"] if trajectory and "id" in trajectory else str(uuid.uuid4())
    log_prefix = f"[Serper Search Request ID: {request_id}] "
    assert serper_api_key is not None, "serper_api_key must be provided for Serper API calls"
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": serper_api_key,
    }
    results = []
    last_error = None
    for query in query_list:
        payload = {"q": query, "num": topk}
        try:
            response = requests.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            results.append(data)
        except Exception as e:
            last_error = f"{log_prefix}Serper API error: {e}"
            logger.error(last_error)
            return None, last_error
    return {"result": results}, None

def call_crawl_api(
    crawl_service_url: str,
    url_list: list[str],
    timeout: int = DEFAULT_TIMEOUT,
    trajectory: dict[str, Any] = None
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    request_id = trajectory["id"] if trajectory and "id" in trajectory else str(uuid.uuid4())
    log_prefix = f"[Crawl Request ID: {request_id}] "
    payload = {"urls": url_list}
    return call_api(crawl_service_url, payload, timeout=timeout, log_prefix=log_prefix)

def _search_passages2string(retrieval_result):
    """Convert retrieval results to formatted string."""
    format_reference = ""
    format_reference += f"Found {len(retrieval_result)} results:\n"
    for idx, doc_item in enumerate(retrieval_result):
        doc = doc_item.get("document", doc_item)
        url = doc.get("url", "No URL provided")
        title = doc.get("title", "No title provided")
        content = doc.get("content") or doc.get("text") or doc.get("snippet") or "No snippet provided"
        score = doc_item.get("score")
        score_line = f"Score: {score}\n" if score is not None else ""
        format_reference += f"{idx + 1}. url: {url}\nTitle: {title}\n{score_line}Snippet: {content}\n\n"
    return format_reference.strip()

def _serper_search_passages2string(retrieval_result):
    """Convert retrieval results to formatted string."""
    organic_results = retrieval_result.get("organic", [])

    if not organic_results:
        return "No organic results found for this query."

    format_reference = ""
    format_reference += f"Found {len(organic_results)} results:\n"
    for idx, doc_item in enumerate(organic_results):
        url = doc_item.get("link", "No URL provided")
        title = doc_item.get("title", "No Title provided")
        content = doc_item.get("snippet", "No Snippet provided")
        format_reference += f"{idx + 1}. url: {url}\nTitle: {title}\nSnippet: {content}\n\n"
    return format_reference.strip()

def _crawl_passages2string(crawl_result):
    """Convert crawl results to formatted string from the retrieval server."""
    url = crawl_result.get("url", "No URL provided")
    result = crawl_result.get("result", "No result available.")
    texts = crawl_result.get("texts", [])

    lines = [f"url: {url}", f"Result: {result}"]
    if texts:
        lines.append(f"Passages: {len(texts)}")
        for idx, text in enumerate(texts, start=1):
            lines.append(f"Passage {idx}: {text}")
    return "\n".join(lines).strip()

def _perform_single_batch_operation(
    operation_name: str,
    item_list: List[str],
    api_call_func: Callable[..., tuple[Optional[dict], Optional[str]]],
    api_call_kwargs: Dict[str, Any],
    result_formatter_func: Callable[[Any], str],
    metadata_fields: Dict[str, str],
    concurrent_semaphore: Optional[threading.Semaphore] = None,
) -> tuple[str, dict[str, Any]]:
    """
    Performs a generic single batch API operation (search or crawl).

    Args:
        operation_name: The name of the operation (e.g., "search", "crawl") for logging.
        item_list: The list of items to process (e.g., queries, URLs).
        api_call_func: The function to call the specific API (e.g., call_search_api).
        api_call_kwargs: Keyword arguments for the API call function.
        result_formatter_func: The function to format the raw API results into a string.
        metadata_fields: A dictionary mapping generic keys to specific keys for metadata
                         (e.g., {"count_key": "query_count", "list_key": "queries"}).
        concurrent_semaphore: Optional semaphore for concurrency control.

    Returns:
        A tuple (result_text, metadata).
        result_text: The operation result JSON string.
        metadata: Metadata dictionary for the batch operation.
    """
    logger.info(f"Starting batch {operation_name} for {len(item_list)} items.")

    api_response = None
    error_msg = None
    req_begin_time = datetime.now().isoformat()

    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = api_call_func(**api_call_kwargs)
        else:
            api_response, error_msg = api_call_func(**api_call_kwargs)
    except Exception as e:
        error_msg = f"API Request Exception during batch {operation_name}: {e}"
        logger.error(f"Batch {operation_name}: {error_msg}")
        traceback.print_exc()

    metadata = {
        metadata_fields["count_key"]: len(item_list),
        metadata_fields["list_key"]: item_list,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "total_results": 0,
        "formatted_result": None,
        "req_begin_time": req_begin_time,
    }

    result_text = json.dumps({"result": f"{operation_name.capitalize()} request failed or timed out after retries."}, ensure_ascii=False)

    if error_msg:
        metadata["status"] = "api_error"
        result_text = json.dumps({"result": f"{operation_name.capitalize()} error: {error_msg}"}, ensure_ascii=False)
        logger.error(f"Batch {operation_name}: API error occurred: {error_msg}")
    elif api_response:
        logger.debug(f"Batch {operation_name}: API Response: {api_response}")
        metadata["api_response"] = api_response

        try:
            if "result" in api_response and api_response["result"]:
                raw_results = api_response["result"]
            elif "results" in api_response and api_response["results"]:
                raw_results = api_response["results"]
            else:
                raw_results = []

            if raw_results:
                pretty_results = []
                total_results = 0
                empty_results = []

                for idx, retrieval in enumerate(raw_results):
                    if operation_name != "crawl" and len(retrieval) == 0 and item_list and idx < len(item_list):
                        empty_results.append({"index": idx, "item": item_list[idx]})
                    formatted = result_formatter_func(retrieval)
                    pretty_results.append(formatted)
                    total_results += len(retrieval) if isinstance(retrieval, list) else 1

                if empty_results:
                    logger.warning(f"Batch {operation_name}: Empty results for items: {empty_results}, response: {api_response}")
                final_result = "\n---\n".join(pretty_results)
                result_text = json.dumps({"result": final_result}, ensure_ascii=False)
                metadata["status"] = "success"
                metadata["total_results"] = total_results
                metadata["formatted_result"] = final_result
                logger.info(f"Batch {operation_name}: Successful, got {total_results} total results")
            else:
                result_text = json.dumps({"result": f"No {operation_name} results found."}, ensure_ascii=False)
                metadata["status"] = "no_results"
                metadata["total_results"] = 0
                logger.warning(f"Batch {operation_name}: No results found")
        except Exception as e:
            error_msg = f"Error processing {operation_name} results: {e}"
            result_text = json.dumps({"result": error_msg}, ensure_ascii=False)
            metadata["status"] = "processing_error"
            logger.error(f"Batch {operation_name}: {error_msg}")
    else:
        metadata["status"] = "unknown_api_state"
        result_text = json.dumps(
            {"result": "Unknown API state (no response and no error message)."}, ensure_ascii=False
        )
        logger.error(f"Batch {operation_name}: Unknown API state.")

    return result_text, metadata

def perform_single_search_batch(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    timeout: int = DEFAULT_TIMEOUT,
    use_serper_api: bool = False,
    serper_api_key: str = None,
    trajectory: dict[str, Any] = None
) -> tuple[str, dict[str, Any]]:
    """
    Performs a single batch search for multiple queries (original search tool behavior).

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        concurrent_semaphore: Optional semaphore for concurrency control.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (result_text, metadata).
        result_text: The search result JSON string.
        metadata: Metadata dictionary for the batch search.
    """
    metadata_fields = {"count_key": "query_count", "list_key": "queries"}

    if use_serper_api:
        api_call_kwargs = {
            "retrieval_service_url": retrieval_service_url,
            "query_list": query_list,
            "topk": topk,
            "timeout": timeout,
            "serper_api_key": serper_api_key,
            "trajectory": trajectory,
        }
        return _perform_single_batch_operation(
            operation_name="serper_search",
            item_list=query_list,
            api_call_func=call_serper_api_direct,
            api_call_kwargs=api_call_kwargs,
            result_formatter_func=_serper_search_passages2string,
            metadata_fields=metadata_fields,
            concurrent_semaphore=concurrent_semaphore,
        )
    else:
        api_call_kwargs = {
            "retrieval_service_url": retrieval_service_url,
            "query_list": query_list,
            "topk": topk,
            "return_scores": True,
            "timeout": timeout,
            "trajectory": trajectory,
        }

        return _perform_single_batch_operation(
            operation_name="search",
            item_list=query_list,
            api_call_func=call_search_api,
            api_call_kwargs=api_call_kwargs,
            result_formatter_func=_search_passages2string,
            metadata_fields=metadata_fields,
            concurrent_semaphore=concurrent_semaphore,
        )

def perform_single_crawl_batch(
    crawl_service_url: str,
    url_list: list[str],
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    timeout: int = DEFAULT_TIMEOUT,
    trajectory: dict[str, Any] = None
) -> tuple[str, dict[str, Any]]:
    """Fetch raw passages for a batch of URLs in a single API call."""
    logger.info(f"Starting batch crawl for {len(url_list)} urls.")
    req_begin_time = datetime.now().isoformat()

    api_call_kwargs = {
        "crawl_service_url": crawl_service_url,
        "url_list": url_list,
        "timeout": timeout,
        "trajectory": trajectory,
    }

    api_response = None
    error_msg = None
    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = call_crawl_api(**api_call_kwargs)
        else:
            api_response, error_msg = call_crawl_api(**api_call_kwargs)
    except Exception as e:
        error_msg = f"API Request Exception during crawl: {e}"
        logger.error("Batch crawl: %s", error_msg)
        traceback.print_exc()

    if error_msg:
        result_text = json.dumps({"result": f"Crawl error: {error_msg}"}, ensure_ascii=False)
        metadata = {
            "url_count": len(url_list),
            "urls": url_list,
            "api_request_error": error_msg,
            "status": "api_error",
            "total_results": 0,
            "formatted_result": None,
            "req_begin_time": req_begin_time,
        }
        return result_text, metadata

    crawl_results = api_response.get("results", []) if api_response else []
    total_results = sum(len(r.get("texts", [])) for r in crawl_results)

    pretty_results = [_crawl_passages2string(result) for result in crawl_results]
    final_result = "\n---\n".join(pretty_results) if pretty_results else "No crawl results found."
    result_text = json.dumps({"result": final_result}, ensure_ascii=False)

    metadata = {
        "url_count": len(url_list),
        "urls": url_list,
        "api_request_error": None,
        "api_response": crawl_results,
        "status": "success" if total_results > 0 else "no_results",
        "total_results": total_results,
        "formatted_result": final_result,
        "req_begin_time": req_begin_time,
    }
    return result_text, metadata