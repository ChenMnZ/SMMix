python -m torch.distributed.launch \
--nproc_per_node=1 \
--use_env  \
--master_port 29533 \
main.py  \
--dist-eval \
--eval \
"$@"