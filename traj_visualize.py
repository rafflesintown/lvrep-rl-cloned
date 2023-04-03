import numpy as np
import torch
import gym
import argparse
import os

import matplotlib.pyplot as plt


# def main():


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--tr_seed", default=0, type=int)              # specifies which training seed to evalute
  parser.add_argument("--rf_num", default = 512, type = int)
  parser.add_argument("--eval_seed", default = 1111)
  parser.add_argument("--learn_rf", default = "False") #string indicating if learn_rf is false or no
  args = parser.parse_args()
  filename = f"traj_log/traj_rf_num={args.rf_num}_learn_rf={args.learn_rf}_tr_seed={args.tr_seed}_eval_seed={args.eval_seed}_euler=False_sigma=0.0.npy"
  # os.makedirs(os.path.dirname(filename), exist_ok = True)
  with open(filename, 'rb') as f:
    all_traj = np.load(f)
    dist_to_opt = np.empty(all_traj.shape[1])
    print(all_traj.shape, "shape of data")
    for i in np.arange(all_traj.shape[1]):
      th = np.arctan2(all_traj[0,i,1], all_traj[0,i,0])
      th_norm = ((th + np.pi) % (2 * np.pi)) - np.pi
      thdot = all_traj[0,i,2]
      dist_to_opt[i] = np.sqrt(th_norm**2)
    plt.plot(np.arange(all_traj.shape[1]), dist_to_opt)  
    plt.savefig("sdec/traj_log/sdec_performance.pdf")



