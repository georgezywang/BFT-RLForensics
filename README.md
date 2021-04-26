# Identifier and Attacker Game for PBFT

## Setup and Running
To set up the prerequisites including Tensorboard visualization, run:
```bash
pip install -r requirements.txt
pip install tensorboard
```
To train agents with cpu，run:
```
python3 src/main.py --config=separate_actor_critic_cpu --env-config=pbft
```
If cuda, run:
```
python3 src/main.py --config=separate_actor_critic_gpu --env-config=pbft
```
