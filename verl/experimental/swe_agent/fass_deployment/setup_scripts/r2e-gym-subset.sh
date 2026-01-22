#!/bin/bash

export PIP_CACHE_DIR=~/.cache/pip
export PATH=/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ln -s /testbed/.venv /root/.venv
ln -s /testbed/.venv/bin/python /root/.local/bin/python
ln -s /testbed/.venv/bin/python /root/.local/bin/python3
find "/testbed/.venv/bin" -type f -executable -exec ln -sf {} "/root/.local/bin/" \;

uv pip install chardet
uv pip install networkx  # for search tool

find . -name '*.pyc' -delete
find . -name '__pycache__' -exec rm -rf {} +
find /r2e_tests -name '*.pyc' -delete
find /r2e_tests -name '__pycache__' -exec rm -rf {} +

mv /testbed/run_tests.sh /root/run_tests.sh
mv /testbed/r2e_tests /root/r2e_tests

mv /r2e_tests /root/r2e_tests
ln -s /root/r2e_tests /testbed/r2e_tests
