#!/bin/bash

export PIP_CACHE_DIR=~/.cache/pip
export PATH=/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH

ln -s /opt/miniconda3/envs/testbed /root/.venv
python -m pip install chardet
# python -m pip install networkx  # for search tool

