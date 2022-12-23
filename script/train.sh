python -m torch.distributed.launch \
--nproc_per_node=4 \
--use_env  \
--master_port 29533 \
main.py  \
--dist-eval \
"$@"


