# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import os
import sys
from pathlib import Path

from loguru import logger

logger.remove()


debug_mode = os.getenv("DEBUG_MODE", False)
if debug_mode:
    logger.add(sys.stdout, level="INFO")
    logger.add(sys.stderr, level="ERROR")

_handler_registry = {}


def add_file_handler(file_path: Path, run_id: str, level: str = "info"):
    def case_filter(record):
        return record["extra"].get("run_id") == run_id and record["level"].no >= logger.level(level.upper()).no

    handler_id = logger.add(
        str(file_path),
        level=level.upper(),
        filter=case_filter,
        format="{time:YYYY-MM-DD HH:mm:ss} | {extra[name]: <10} | {level: <8} | {message}",
        enqueue=True,
    )
    if run_id not in _handler_registry:
        _handler_registry[run_id] = []
    _handler_registry[run_id].append(handler_id)
    return handler_id


def get_logger(
    name: str,
    run_id: str,
) -> logging.Logger:
    return logger.bind(name=name, run_id=run_id)


def cleanup_handlers(run_id: str):
    if run_id in _handler_registry:
        for handler_id in _handler_registry[run_id]:
            try:
                logger.remove(handler_id)
            except ValueError:
                pass
        del _handler_registry[run_id]
