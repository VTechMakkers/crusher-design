#!/usr/bin/env bash
#
# One-shot setup on the Ryzen 5 7600X + RTX 3090 + RTX 3080 workstation.
# Installs: CUDA-aware Python, vLLM, LIGGGHTS, CalculiX, gmsh, project deps.
#
# Tested on Ubuntu 22.04 LTS. Adapt for other distros.

set -euo pipefail

echo "=== checking NVIDIA driver ==="
nvidia-smi | head -3

echo "=== system packages ==="
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git wget curl \
    python3.11 python3.11-venv python3-pip \
    libopenmpi-dev openmpi-bin \
    libvtk9-dev libboost-all-dev \
    calculix-ccx gmsh

echo "=== Python venv ==="
python3.11 -m venv /opt/crusher-design/.venv
source /opt/crusher-design/.venv/bin/activate
pip install --upgrade pip wheel

echo "=== project deps ==="
pip install \
    "cadquery-ocp>=7.7" \
    "pyyaml>=6.0" \
    "mcp>=1.0" \
    "numpy>=1.26" \
    "httpx>=0.27" \
    "torch>=2.4" \
    "vllm>=0.6"   # check latest compatible with sm_86 (Ampere)

echo "=== LIGGGHTS-PUBLIC ==="
# LIGGGHTS-PUBLIC (CFDEM fork is most active in 2026)
if [ ! -d /opt/LIGGGHTS-PUBLIC ]; then
    sudo git clone --depth=1 https://github.com/CFDEMproject/LIGGGHTS-PUBLIC.git \
        /opt/LIGGGHTS-PUBLIC
fi
cd /opt/LIGGGHTS-PUBLIC/src
make -j$(nproc) auto
sudo cp lmp_auto /usr/local/bin/liggghts
echo "  liggghts → $(which liggghts)"

echo "=== install systemd units ==="
sudo cp /opt/crusher-design/infra/vllm-coder.service /etc/systemd/system/
sudo cp /opt/crusher-design/infra/vllm-fast.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable vllm-coder.service vllm-fast.service

echo "=== smoke test ==="
cd /opt/crusher-design
python3 -m loop.design_loop || true

echo
echo "=== READY ==="
echo "Start LLMs:  sudo systemctl start vllm-coder vllm-fast"
echo "Local coder: http://127.0.0.1:8001"
echo "Local fast:  http://127.0.0.1:8002"
echo "LIGGGHTS:    liggghts -h"
