#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

python -m torch.distributed.launch --nproc_per_node=${NGPUS} test.py --launcher pytorch ${PY_ARGS}
# python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_addr=localhost --master_port ${PORT} train.py --launcher pytorch ${PY_ARGS}
