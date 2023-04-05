#computes the performance of the learned koopman-U model for Pendulum-v1
import sys
sys.path.append("../utility")
sys.path.append("../train")
sys.path.append("../gym_env")

import numpy as np
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import random
import argparse
from collections import OrderedDict
from copy import copy
import scipy
import scipy.linalg
from Utility import data_collecter

import os
from numpy.linalg import inv


import Learn_Koopman_with_KlinearEig as lka
from our_env.noisy_pend import noisyPendulumEnv
from pendulum import PendulumEnv



def Prepare_Region_LQR(env_name, Nstate,NKoopman, thdot_weight=0.1, u_weight = 0.001):
	x_ref = np.zeros(Nstate)
	if env_name == "Pendulum-v1":
		Q = np.zeros((NKoopman,NKoopman))
		Q[0,0] = 1.
		Q[1,1] = thdot_weight
		R = np.eye(1) * u_weight
	return Q,R, x_ref

def Psi_o(s,net,NKoopman): # Evaluates basis functions Psi(s(t_k))
	psi = np.zeros([NKoopman,1])
	ds = net.encode(torch.DoubleTensor(s)).detach().cpu().numpy()
	psi[:NKoopman, 0] = ds
	return psi


def dlqr(A,B,Q,R,gamma, max_iters = 100):
    #iteratively solve P and q
    #returns the optimal K such that u = -Kx
    P=Q
    # q=-Q.dot(x_)
    n = A.shape[0]
    for i in range(max_iters):
        
        t1=gamma*B.T.dot(P).dot(A)
        P_new=Q-t1.T.dot(inv(R+gamma*B.T.dot(P).dot(B))).dot(t1)+gamma*A.T.dot(P).dot(A)
        # q_new=-Q.dot(x_)-t1.T.dot(inv(R+gamma*B.T.dot(P).dot(B))).dot(gamma*B.T.dot(q))+gamma*A.T.dot(q)
        # print(i, np.linalg.norm(P_new-P))
        if np.linalg.norm(P_new-P)<1e-6:
            break
        P=P_new
        # q=q_new
    t2=inv(R+gamma*B.T.dot(P).dot(B))
    K=t2.dot(gamma*B.T.dot(P).dot(A))
    # K_ = (-t2) @ (gamma* B.T) @inv(np.eye(n) - (A + B @ K).T) @ (-Q)
    # g=-t2.dot(gamma*B.T.dot(q))
    return K



def exp(env,env_name,net,Kopt,init_state,x_ref,Nstate,NKoopman,gamma = 0.99,max_steps = 100):
	observation_list = []
	print("init state", init_state)
	observation = np.array(env.reset_koopman(init_state = init_state)) #obs is [normalized th,thdot]
	print("env state now after running reset", env.state)
	x0 = np.matrix(Psi_o(observation,net,NKoopman)).reshape(NKoopman,1)
	x_ref_lift = Psi_o(x_ref,net,NKoopman).reshape(-1,1)
	observation_list.append(x0[:Nstate].reshape(-1,1))
	u_list = []
	# flag = False
	for i in range(max_steps):
		u = np.matmul(-Kopt, (x0 - x_ref_lift))
		u = np.array([u[0,0]])
		observation,reward,done,info = env.step_koopman(u)
		# if steps >= 200:
		# 	flag = True
		x0 = np.matrix(Psi_o(observation,net,NKoopman)).reshape(NKoopman,1)
		observation_list.append(x0[:Nstate].reshape(-1,1))
		u_list.append(u)
		# print("obs at step %d" %i, observation )
	u_list = np.array(u_list).reshape(-1)
	observations = np.concatenate(observation_list,axis =1)
	return observations, u_list


def normalize(angle):
	return ((angle + np.pi) % (2 * np.pi) - np.pi)

def Cost(observations,u_list,Q,R,x_ref,gamma = 0.99):
	steps = observations.shape[1]
	loss = 0
	for s in range(steps):
		if s != steps - 1:
			ucost = 0.001 * u_list[s]**2
			# loss += ucost * gamma**s
			loss += ucost 
		# xcost = np.dot(np.dot((observations[:,s]-x_ref).T,Q),(observations[:,s]-x_ref))
		x_cost = normalize(observations[0,s]) ** 2 + 0.1* observations[1,s] ** 2
		# print("observations s", observations[:,s])
		# loss += x_cost * gamma**s
		loss += x_cost
		# print("loss at step %d" %s, x_cost + ucost)
	return loss

def main():
	method = "KoopmanU"
	suffix = "Pendulum-v1"
	env_name = "Pendulum-v1"
	root_path = "../Data/" + suffix
	return_norm_th = "False"
	# samples = 100000
	# dt = 0.05
	# seed = 1
	for file in os.listdir(root_path):
		wow = f"return_norm_th={return_norm_th}.pth"
		print("wow", wow)
		if file.startswith(method+"_") and file.endswith(f"return_norm_th={return_norm_th}.pth"):
			model_path = file
	Data_collect = data_collecter(env_name)
	udim = Data_collect.udim
	Nstate = Data_collect.Nstates
	print("Nstate",Nstate)
	layer_depth = 3
	layer_width = 128
	dicts = torch.load(root_path + "/" + model_path)
	state_dict = dicts["model"]

	#build net for koopmanU
	layer = dicts["layer"]
	NKoopman = layer[-1] + Nstate
	net = lka.Network(layer,NKoopman,udim)
	net.load_state_dict(state_dict)
	device = torch.device("cpu")
	net.cpu()
	net.double()

	#perform eval experiments
	gamma = 1.
	Ad = state_dict['lA.weight'].cpu().numpy()
	Bd = state_dict['lB.weight'].cpu().numpy()
	Q,R,x_ref = Prepare_Region_LQR(env_name,Nstate,NKoopman)
	Ad = np.matrix(Ad)
	Bd = np.matrix(Bd)
	Kopt = dlqr(Ad,Bd,Q,R,gamma = gamma)


	#try evaluate over 100 init states
	np.random.seed(0)
	max_steps = 200
	n_init_states = 100
	# init_states = np.random.uniform(low = [-np.pi,-1.],
	# 	high = [np.pi, 1.], size = (n_init_states, 2))

	n_init_states = 1
	init_states = np.array([[-3,0]])

	final_costs = np.empty(n_init_states)

	env = PendulumEnv(return_norm_th = return_norm_th)
	for i in np.arange(n_init_states):
		init_state = init_states[i,:]
		observations, u_list = exp(env, env_name, net,Kopt,init_state,x_ref,Nstate,
			NKoopman,max_steps = max_steps, gamma = gamma)
		print("u-list[:10]", u_list[:10])
		cost_i = Cost(observations,u_list,Q,R,x_ref,gamma = gamma)
		final_costs[i] = cost_i
		print(f"final cost for init state {init_state}  is {cost_i}")
		# vis_env = noisyPendulumEnv(dt = dt)
		# vis_env.visualize(init_state = init_states[i], cmd = u_list)

	print("mean cost", np.mean(final_costs))
	# print("final costs", final_costs)




if __name__ == '__main__':
	main()