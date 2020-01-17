
# Instructions
## CRCV Cluster
### Single GPU Training

```
sbatch -p preempt --qos=preempt -c 8 --mem=32G --gres=gpu:turing:1 --wrap="XVIEW_CONFIG=turing_focal_gamma1 python train.py"
```


### Multi GPU Training
```
sbatch -p preempt --qos=preempt -c 32 --mem=160G --gres=gpu:turing:4 --wrap="XVIEW_CONFIG=turing_fixcls python -m torch.distributed.launch --nproc_per_node=4 --master_port=420666 train.py"
```


## Local machine
### Single GPU Training
```
XVIEW_CONFIG=turing_focal_gamma1 python train.py
```

