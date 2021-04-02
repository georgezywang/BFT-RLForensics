"""
Code adapted from https://github.com/TonghanWang/ROMA
"""

import datetime
import os
import pprint
import time
import threading

import torch
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learner import REGISTRY as le_REGISTRY
from runner import REGISTRY as r_REGISTRY
from controller import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    if args.use_cuda:
        th.cuda.set_device(args.device_num)

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        args.latent_role_direc = os.path.join(tb_exp_direc, "{}").format('latent_role')
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_distance_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_distance_sequential(args, logger):
    # meta-train the actor agents
    #   - generate task distributions ()
    #   - train actor agents + sampler
    #       - for i
    #   - save the trained model
    # train distance z, z' [-1, 1]
    #   -
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    # args.state_shape = env_info["state_shape"]

    #    args.own_feature_size = env_info["own_feature_size"] #unit_type_bits+shield_bits_ally
    # if args.obs_last_action:
    #    args.own_feature_size+=args.n_actions
    # if args.obs_agent_id:
    #    args.own_feature_size+=args.n_agents

    # Default/Base scheme
    scheme = {
        # "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "z_q": {"vshape": (args.latent_relation_space_dim,), "group": "agents", "dtype": th.float},
        "z_p": {"vshape": (args.latent_relation_space_dim,), "group": "agents", "dtype": th.float},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "rewards": {"vshape": env_info["reward_shape"], },
        "redistributed_rewards": {"vshape": env_info["reward_shape"], },
        "adjacent_agents": {"vshape": env_info["adjacent_agents_shape"], "group": "agents", },
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning meta-training for {} timesteps".format(args.total_pretrain_steps))

    tasks = generate_dist_distributions(args)
    while runner.t_env <= args.total_pretrain_steps:
        for z_q, z_p in tasks:
            # Run for a whole episode at a time
            episode_batch = runner.run(z_q, z_p, test_mode=False)
            buffer.insert_episode_batch(episode_batch)

            if buffer.can_sample(args.batch_size):
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, episode)

            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

    logger.console_logger.info("Beginning training for {} timesteps".format(args.total_z_training_steps*args.env_steps_every_z))
    runner.t_env = 0
    z_train_cnt = 0
    env_steps_per_z = 0
    z_p, z_q = generate_dist_distributions(args, num=1)
    z_optimiser = torch.optim.Adam(params=[z_p, z_q], lr=args.z_update_lr, eps=args.optim_eps)
    while z_train_cnt <= args.total_z_training_steps:
        device = "cpu" if args.buffer_cpu_only else args.device
        buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                              preprocess=preprocess,
                              device=device)

        env_steps_per_z += args.env_steps_every_z
        while runner.t_env <= env_steps_per_z:
            # Run for a whole episode at a time
            episode_batch = runner.run(z_q, z_p, test_mode=False)
            buffer.insert_episode_batch(episode_batch)

            if buffer.can_sample(args.batch_size):
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, episode)

            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

        episode_returns = []
        for _ in range(args.z_sample_runs):
            episode_returns.append(runner.run(z_q, z_p, test_mode=True, sample_mode=True))
        data = {"z_p": z_p, "z_q": z_q, "evals": episode_returns}
        train_batch = {}
        for k, v in data.items():
            if not isinstance(v, th.Tensor):
                v = th.tensor(v, dtype=th.long, device=device)
            else:
                v.to(device)
            train_batch.update({k: v})
        # train z critic
        train_batch["evals"] = torch.sum(train_batch["evals"], dim=0) / args.z_sample_runs
        learner.z_train(train_batch, device, z_train_cnt)

        # update z_q, z_p
        total_val = learner.get_social_welfare_z(train_batch, device)
        z_optimiser.zero_grad()
        total_val.backward()
        z_optimiser.step()

        z_train_cnt += 1

        t_max = args.env_steps_every_z * args.total_z_training_steps
        logger.log_stat("avg_social_welfare", (total_val.item()), runner.t_env)
        logger.console_logger.info("t_env: {} / {}".format(runner.t_env, t_max))
        logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
            time_left(last_time, last_test_T, runner.t_env, t_max), time_str(time.time() - start_time)))
        last_time = time.time()

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(z_q, z_p, test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")



def generate_dist_distributions(args, num=None):
    # [(z_q, z_p)]: z_q: [n_agents][space_dim]
    # FIXME: any better (more spreading) way than uniform?
    distribution = torch.distributions.uniform.Uniform(torch.tensor([args.latent_relation_space_lower_bound], dtype=torch.float),
                                                       torch.tensor([args.latent_relation_space_upper_bound], dtype=torch.float))

    if num is None:
        tasks = [(distribution.sample(torch.Size([args.n_agents, args.latent_relation_space_dim])).view(1, args.n_agents, args.latent_relation_space_dim),
                  distribution.sample(torch.Size([args.n_agents, args.latent_relation_space_dim])).view(1, args.n_agents, args.latent_relation_space_dim))
                 for _ in range(args.pretrained_task_num)]
    else:
        tasks = (distribution.sample(torch.Size([args.n_agents, args.latent_relation_space_dim])).view(args.n_agents, args.latent_relation_space_dim),
                 distribution.sample(torch.Size([args.n_agents, args.latent_relation_space_dim])).view(args.n_agents, args.latent_relation_space_dim))
        tasks[0].requires_grad = True
        tasks[1].requires_grad = True
    return tasks

def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config
