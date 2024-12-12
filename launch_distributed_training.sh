#!/bin/bash

MASTER_IP="129.104.253.36"
PORT="29502"
NNODES=6
NPROC_PER_NODE=1
BATCH_SIZE=6
SNAPSHOT_PATH=/users/eleves-a/2022/amine.chraibi/difftransformer.pt

ssh dindon.polytechnique.fr "python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --node_rank=0 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_IP}:${PORT} KaggleINF/diff_transformer_trainer.py 100 5 --batch_size ${BATCH_SIZE}"&
ssh malleole.polytechnique.fr "python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --node_rank=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_IP}:${PORT} KaggleINF/diff_transformer_trainer.py 100 5 --batch_size ${BATCH_SIZE}" > /dev/null 2>&1&
ssh kamiche.polytechnique.fr "python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --node_rank=2 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_IP}:${PORT} KaggleINF/diff_transformer_trainer.py 100 5 --batch_size ${BATCH_SIZE}" > /dev/null 2>&1&
ssh loriol.polytechnique.fr "python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --node_rank=3 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_IP}:${PORT} KaggleINF/diff_transformer_trainer.py 100 5 --batch_size ${BATCH_SIZE}" > /dev/null 2>&1&
ssh quetzal.polytechnique.fr "python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --node_rank=4 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_IP}:${PORT} KaggleINF/diff_transformer_trainer.py 100 5 --batch_size ${BATCH_SIZE}" > /dev/null 2>&1&
ssh jabiru.polytechnique.fr "python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --node_rank=5 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_IP}:${PORT} KaggleINF/diff_transformer_trainer.py 100 5 --batch_size ${BATCH_SIZE}" > /dev/null 2>&1&

