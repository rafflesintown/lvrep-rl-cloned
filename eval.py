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

from agent.rfsac.rfsac_agent import RFVCritic, nystromVCritic
from agent.sac.actor import DiagGaussianActor
from agent.rfsac.rfsac_agent import RFSACAgent


def main(env,log_path,agent,rf_num, learn_rf, max_steps = 200, tr_seed = 0, true_state_dim = None, save_traj = False, euler = False, sigma = 0.0):
  # eval_seed = 100
  # eval_seed = 0
  # eval_seed = 1000
  eval_seed = 1000
  # eval_seed = 2023
  np.random.seed(eval_seed)
  n_init_states = 100
  low_arr =  [-np.pi,-1.]
  high_arr = [np.pi, 1.]
  init_states = np.random.uniform(low = low_arr, high = high_arr, size = (n_init_states,true_state_dim))
  # n_init_states = 1
  # init_states = [np.array([2,0.0])]
  all_rewards = np.empty(n_init_states)
  u_list = np.empty(max_steps)
  obs_dim  = env.observation_space.shape[0]
  a_dim = 1
  if save_traj == True:
    all_traj = np.empty((n_init_states, max_steps,obs_dim+a_dim))


  for i in np.arange(n_init_states):
    env_seed = 100 #seed for noise
    # nv_seed = 100 #seed for noise
    init_state = init_states[i]
    state = env.reset(init_state = init_state)
    eps_reward = 0
    # np.random.seed(env_seed)
    for t in range(max_steps):
      # print("current state", state)
      # print("time %d" %t)
      # print("previous state", state)
      action = agent.select_action(np.array(state))
      if save_traj == True:
        all_traj[i,t,:obs_dim] = state
        all_traj[i,t, obs_dim:] = action
      # print("previous action", action)
      # print("current action", action)
      state,reward,done, _ = env.step(action)
      # print("current state", state)
      # print("current reward", reward)
      eps_reward += reward
      u_list[t] = action
    all_rewards[i] = eps_reward
    # env.visualize(init_state = init_states[i], cmd = u_list, seed = env_seed) 

    # print("eval eps reward for init state: ", init_state,  ": ", eps_reward  )
  
  print(f"mean episodic reward over {max_steps} time steps (rf_num = {rf_num}, learn_rf = {learn_rf}, seed ={tr_seed}, euler = {euler}, sigma = {sigma}, use_nystrom = {use_nystrom}): ", np.mean(all_rewards))
  if save_traj == True:
    filename = f"traj_log/traj_rf_num={rf_num}_learn_rf={learn_rf}_tr_seed={tr_seed}_eval_seed={eval_seed}_euler={euler}_sigma={sigma}_use_nystrom={use_nystrom}.npy"
    os.makedirs(os.path.dirname(filename), exist_ok = True)
    with open(filename, 'wb') as f:
      np.save(f, all_traj)

  filename = f"cost_log/cost_rf_num={rf_num}_learn_rf={learn_rf}_tr_seed={tr_seed}_eval_seed={eval_seed}_euler={euler}_sigma={sigma}_use_nystrom={use_nystrom}.npy"
  os.makedirs(os.path.dirname(filename), exist_ok = True)
  with open(filename, 'wb') as f:
    np.save(f, all_rewards)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default=0, type=int)                     
  parser.add_argument("--alg", default="rfsac")                     # Alg name (sac, vlsac,rfsac)
  parser.add_argument("--env", default="Pendulum-v1")          # Environment name
  parser.add_argument("--tr_seed", default=0, type=int)              # specifies which training seed to evalute
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
  parser.add_argument("--rf_num", default = 512, type = int)
  parser.add_argument("--learn_rf", default = "False") #string indicating if learn_rf is false or no
  parser.add_argument("--save_traj", default = "False") #string for if want to save {[init_state, action_list]} or not
  parser.add_argument("--euler", default = "False") 
  parser.add_argument("--use_nystrom", default = "False")
  args = parser.parse_args()
  

  sigma = args.sigma
  euler = True if args.euler == "True" else False
  env = gym.make(args.env)
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0] 
  max_action = float(env.action_space.high[0])
  max_length = env._max_episode_steps
  print("hey", args.learn_rf)
  learn_rf = True if args.learn_rf == "True" else False
  save_traj = True if args.save_traj == "True" else False
  use_nystrom = True if args.use_nystrom == "True" else False
  kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "action_space": env.action_space,
    "discount": args.discount,
    "tau": args.tau,
    "hidden_dim": args.hidden_dim,
    'sigma': sigma,
    "rf_num": args.rf_num,
    "learn_rf": learn_rf,
    "use_nystrom": use_nystrom
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
    # env = noisyPendulumEnv(sigma =  sigma, euler = euler)
    env = noisyPendulumEnv(sigma = sigma, euler = euler)
    if use_nystrom == True:
      log_path = f'log/{args.env}/{args.alg}/{args.dir}/{args.tr_seed}/T={args.max_timesteps}/rf_num={args.rf_num}/learn_rf={learn_rf}/sigma={sigma}/euler={euler}/use_nystrom={use_nystrom}'
    else:
      log_path = f'log/{args.env}/{args.alg}/{args.dir}/{args.tr_seed}/T={args.max_timesteps}/rf_num={args.rf_num}/learn_rf={learn_rf}/sigma={sigma}/euler={euler}'
    # log_path = f'log/{args.env}/{args.alg}/{args.dir}/{args.tr_seed}/T={args.max_timesteps}/rf_num={args.rf_num}/learn_rf={learn_rf}/sigma={sigma}'
    actor = DiagGaussianActor(obs_dim = 3, action_dim = 1,hidden_dim = args.hidden_dim, hidden_depth = 2, 
      log_std_bounds=[-5.,2.])
    if use_nystrom == False:
      critic = RFVCritic(sigma = sigma, rf_num = args.rf_num, learn_rf = learn_rf)
    else:
      critic = nystromVCritic(sigma = sigma, feat_num = args.rf_num)
    actor.load_state_dict(torch.load(log_path+"/actor.pth"))
    critic.load_state_dict(torch.load(log_path + "/critic.pth"))
    agent.actor = actor
    agent.critic = critic

    # for name,param in actor.named_parameters():
    #   print(name, param.data[:1])

    # print("this is critic fourier2 weight first 10", critic.fourier2.state_dict()['weight'][:10])

    # print("this is critic's embedding layer weight", critic.embed.state_dict()['weight'])
    # print("this is critic's embedding layer bias", critic.embed.state_dict()['bias'])

    main(env, log_path,agent, rf_num = args.rf_num, learn_rf = learn_rf, true_state_dim = 2, tr_seed = args.tr_seed, save_traj = save_traj,euler = euler,sigma=sigma)
  
