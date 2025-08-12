#!/usr/bin/env bash
# ---------- [1] 공통 환경 ----------
export REQUESTS_CA_BUNDLE=$(python -m certifi)
export SSL_CERT_FILE=$REQUESTS_CA_BUNDLE
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy=storage.googleapis.com,api.wandb.ai
export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_DISABLE_CODE=true
export WANDB_IGNORE_GLOBS="*.log,*.pth"
export WANDB_HTTP_TIMEOUT=90
export WANDB__SERVICE_WAIT=300
export WANDB__EXECUTABLE=$(which python)

wandb sweep sweep.yaml