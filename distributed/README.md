## Running the code

Key parameters:

- The `--nnodes` flag represents the number of machines communicating over a (simulated) network
- The `--nproc_per_node` flag stands for the number of GPUs on a single machine.

To simulate 2 GPUs (processes) on a single machine (node) run:

```bash
cd distributed
# Start the containers
docker-compose -f docker-compose-multi-process.yaml up -d

# Enter the container and login to wandb
docker-compose -f docker-compose-multi-process.yaml exec node bash
wandb login

# Single node, simulate two GPUs using two processes.
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=node:48123 train.py --no_checkpoint
```

Also, you can simulate training over multiple machines with a single GPU for each:

```bash
cd distributed
# Start the containers
docker-compose -f docker-compose-multi-node.yaml up -d

# Open terminal 1 and login to wandb:
docker-compose -f docker-compose-multi-node.yaml exec node0 bash
wandb login

# Open terminal 2
# Note: the wandb query is not required for node1, as we log data through node0
docker-compose -f docker-compose-multi-node.yaml exec node1 bash

# Run the training script in each terminal
torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=node0:48123 train.py --no_checkpoint
```