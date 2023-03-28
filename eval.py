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

from agent.rfsac.rfsac_agent import RFVCritic
from agent.sac.actor import DiagGaussianActor
from agent.rfsac.rfsac_agent import RFSACAgent


def main(env,log_path,agent,rf_num, learn_rf, max_steps = 200,state_dim = None):
  np.random.seed(0)
  # n_init_states = 100
  # low_arr =  [-np.pi,-1.]
  # high_arr = [np.pi, 1.]
  # init_states = np.random.uniform(low = low_arr, high = high_arr, size = (n_init_states,state_dim))
  n_init_states = 1
  init_states = [np.array([-0.5,0])]
  all_rewards = np.empty(n_init_states)
  u_list = np.empty(max_steps)
  for i in np.arange(n_init_states):
    init_state = init_states[i]
    state = env.reset(init_state = init_state)
    eps_reward = 0
    for t in range(max_steps):
      print("current state", state)
      action = agent.select_action(np.array(state))
      print("current action", action)
      state,reward,done, _ = env.step(action)
      eps_reward += reward
      u_list[t] = action
    all_rewards[i] = eps_reward
    env.visualize(init_state = init_states[i], cmd = u_list)

    # print("eval eps reward for init state: ", init_state,  ": ", eps_reward  )
  
  print(f"mean episodic reward over 200 time steps (rf_num = {rf_num}, learn_rf = {learn_rf}): ", np.mean(all_rewards))





if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default=0, type=int)                     
  parser.add_argument("--alg", default="rfsac")                     # Alg name (sac, vlsac,rfsac)
  parser.add_argument("--env", default="Pendulum-v1")          # Environment name
  parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
  parser.add_argument("--start_timesteps", default=25e3, type=float)# Time steps initial random policy is used
  parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
  parser.add_argument("--max_timesteps", default=1e5, type=float)   # Max time steps to run environment
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
  parser.add_argument("--rand_feat_num", default = 512, type = int)
  parser.add_argument("--learn_rf", default = "False") #string indicating if learn_rf is false or no
  args = parser.parse_args()
  

  sigma = args.sigma
  env = gym.make(args.env)
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0] 
  max_action = float(env.action_space.high[0])
  max_length = env._max_episode_steps
  print("hey", args.learn_rf)
  learn_rf = True if args.learn_rf == "True" else False
  kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "action_space": env.action_space,
    "discount": args.discount,
    "tau": args.tau,
    "hidden_dim": args.hidden_dim,
    "sigma": sigma,
    "rand_feat_num": args.rand_feat_num,
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
  
  if args.env == "Pendulum-v1":
    env = noisyPendulumEnv(sigma =  sigma)
    log_path = f'log/{args.env}/{args.alg}/{args.dir}/{args.seed}/T={args.max_timesteps}/rf_num={args.rand_feat_num}/learn_rf={learn_rf}'
    actor = DiagGaussianActor(obs_dim = 3, action_dim = 1,hidden_dim = args.hidden_dim, hidden_depth = 2, 
      log_std_bounds=[-5.,2.])
    critic = RFVCritic(sigma = sigma, rand_feat_num = args.rand_feat_num, learn_rf = learn_rf)
    actor.load_state_dict(torch.load(log_path+"/actor.pth"))
    critic.load_state_dict(torch.load(log_path + "/critic.pth"))
    agent.actor = actor
    agent.critic = critic

    # print("this is critic's embedding layer weight", critic.embed.state_dict()['weight'])
    # print("this is critic's embedding layer bias", critic.embed.state_dict()['bias'])

    main(env, log_path,agent, rf_num = args.rand_feat_num, learn_rf = learn_rf, state_dim = state_dim)
	

