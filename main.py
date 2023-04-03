import numpy as np
import torch
import gym
import argparse
import os

from tensorboardX import SummaryWriter

from utils import util, buffer
from agent.sac import sac_agent
from agent.vlsac import vlsac_agent
from agent.rfsac import rfsac_agent

from our_env.noisy_pend import noisyPendulumEnv


if __name__ == "__main__":
	
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default=0, type=int)                     
  parser.add_argument("--alg", default="sac")                     # Alg name (sac, vlsac)
  parser.add_argument("--env", default="Pendulum-v1")          # Environment name
  parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
  parser.add_argument("--start_timesteps", default=25e3, type=float)# Time steps initial random policy is used
  parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
  parser.add_argument("--max_timesteps", default=1e6, type=float)   # Max time steps to run environment
  parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
  parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
  parser.add_argument("--hidden_dim", default=256, type=int)      # Network hidden dims
  parser.add_argument("--feature_dim", default=256, type=int)      # Latent feature dim
  parser.add_argument("--discount", default=0.99)                 # Discount factor
  parser.add_argument("--tau", default=0.005)                     # Target network update rate
  parser.add_argument("--learn_bonus", action="store_true")        # Save model and optimizer parameters
  parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
  parser.add_argument("--extra_feature_steps", default=3, type=int)
  parser.add_argument("--sigma", default = 0.,type = float) #noise for noisy environment
  parser.add_argument("--embedding_dim", default = -1,type =int) #if -1, do not add embedding layer
  parser.add_argument("--rf_num", default = 256, type = int)
  parser.add_argument("--learn_rf", default = "False") #make this a string (strange Python issue...) 
  parser.add_argument("--euler", default = "False") #True if euler discretization to be used; otherwise use default OpenAI gym discretization
  args = parser.parse_args()

  sigma = args.sigma
  euler = True if args.euler == "True" else False

  env = gym.make(args.env)
  eval_env = gym.make(args.env)
  max_length = env._max_episode_steps
  if args.env == "Pendulum-v1":
    env = noisyPendulumEnv(sigma =  sigma, euler = euler)
    eval_env = noisyPendulumEnv(sigma = sigma, euler = euler)
  env.seed(args.seed)
  eval_env.seed(args.seed)
  

  # setup log 
  log_path = f'log/{args.env}/{args.alg}/{args.dir}/{args.seed}/T={args.max_timesteps}/rf_num={args.rf_num}/learn_rf={args.learn_rf}/sigma={args.sigma}/euler={euler}'
  summary_writer = SummaryWriter(log_path+"/summary_files")

  # set seeds
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # # 
  # if args.env == "Pendulum-v1": #hard-code for now
  #   state_dim = 2
  #   # print("I am here!")
  # else:
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0] 
  max_action = float(env.action_space.high[0])

  if args.learn_rf == "False":
    learn_rf = False
  else:
    learn_rf = True

  kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "action_space": env.action_space,
    "discount": args.discount,
    "tau": args.tau,
    "hidden_dim": args.hidden_dim,
    "sigma": sigma,
    "rf_num": args.rf_num,
    "learn_rf": learn_rf
  }

  # Initialize policy
  if args.alg == "sac":
    agent = sac_agent.SACAgent(**kwargs)
  elif args.alg == 'vlsac':
    kwargs['extra_feature_steps'] = args.extra_feature_steps
    kwargs['feature_dim'] = args.feature_dim
    agent = vlsac_agent.VLSACAgent(**kwargs)
  elif args.alg == 'rfsac':
    agent = rfsac_agent.RFSACAgent(**kwargs)
  
  replay_buffer = buffer.ReplayBuffer(state_dim, action_dim)

  # Evaluate untrained policy
  evaluations = [util.eval_policy(agent, eval_env)]

  state, done = env.reset(), False
  episode_reward = 0
  episode_timesteps = 0
  episode_num = 0
  timer = util.Timer()

  #keep track of best eval model's state dict
  best_eval_reward = -1e6
  best_actor = None
  best_critic = None

  for t in range(int(args.max_timesteps)):
    
    episode_timesteps += 1

    # Select action randomly or according to policy
    if t < args.start_timesteps:
      action = env.action_space.sample()
    else:
      action = agent.select_action(state, explore=True)



    # Perform action
    next_state, reward, done, _ = env.step(action) 
    # print("next state", next_state)
    done_bool = float(done) if episode_timesteps < max_length else 0

    # if episode_timesteps >= 2: #start adding to buffer once 2 state action pairs seen
    # # Store data in replay buffer
    #   replay_buffer.add(prev_state, prev_action, prev_reward, state, action,reward,next_state, done_bool)
    #   print("prev state", prev_state)
    #   print("prev action", prev_action)
    #   print("state", state)
    #   print("action",action)
    #   print("next state", next_state)
    #   print("prev_reward", prev_reward)
    #   print("reward", reward)

    replay_buffer.add(state,action,next_state,reward,done)

    prev_state = np.copy(state)
    # # print("I am prev state", prev_state)
    # prev_action = np.copy(action)
    # prev_reward = np.copy(reward)
    state = next_state
    # print(" I am state", state)
    episode_reward += reward
    
    # Train agent after collecting sufficient data
    if t >= args.start_timesteps:
      info = agent.train(replay_buffer, batch_size=args.batch_size)

    if done: 
      # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
      print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
      # Reset environment
      state, done = env.reset(), False
      # prev_state = np.copy(state)
      episode_reward = 0
      episode_timesteps = 0
      episode_num += 1 

    # Evaluate episode
    if (t + 1) % args.eval_freq == 0:
      steps_per_sec = timer.steps_per_sec(t+1)
      evaluation = util.eval_policy(agent, eval_env)
      evaluations.append(evaluation)

      if t >= args.start_timesteps:
        info['evaluation'] = evaluation
        for key, value in info.items():
          summary_writer.add_scalar(f'info/{key}', value, t+1)
        summary_writer.flush()

      print('Step {}. Steps per sec: {:.4g}.'.format(t+1, steps_per_sec))

      if evaluation > best_eval_reward:
        best_actor = agent.actor.state_dict()
        best_critic = agent.critic.state_dict()

        # print("actor's state dict")
        # for param_tensor in agent.actor.state_dict():
        #   print(param_tensor, "\t", agent.actor.state_dict()[param_tensor].size())

        # print("critic's state dict")
        # for param_tensor in agent.critic.state_dict():
        #   print(param_tensor,"\t", agent.critic.state_dict()[param_tensor].size())

  summary_writer.close()

  print('Total time cost {:.4g}s.'.format(timer.time_cost()))

  #save best actor/best critic
  torch.save(best_actor, log_path+"/actor.pth")
  torch.save(best_critic, log_path+"/critic.pth")
