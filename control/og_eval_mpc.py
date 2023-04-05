#computes the performance of the learned koopman-U model for Pendulum-v1
import sys
sys.path.append("../utility")
sys.path.append("../train")
sys.path.append("../gym_env")
import do_mpc

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
	print("s shape", s.shape)
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
	observation = np.array(env.reset_state(state = init_state)) #obs is [th,thdot]
	print("curr obs", observation)
	print("env state now after running reset", env.state)
	x0 = np.matrix(Psi_o(observation,net,NKoopman)).reshape(NKoopman,1)
	x_ref_lift = Psi_o(x_ref,net,NKoopman).reshape(-1,1)
	observation_list.append(x0[:Nstate].reshape(-1,1))
	u_list = []
	# flag = False
	for i in range(max_steps):
		u = np.matmul(-Kopt, (x0 - x_ref_lift))
		u = np.array([u[0,0]])
		u = np.minimum(np.maximum(u,-2),2)
		observation,reward,done,info = env.step(u)
		observation = observation.reshape(-1)
		print("observation", observation)
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
			print("u now", u_list[s])
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

def main(tr_seed = 0, save_traj = False, samples = 80000, sigma = 0.0, euler = False):
	method = "KoopmanU"
	suffix = "Pendulum-v1"
	env_name = "Pendulum-v1"
	root_path = "../Data/" + suffix
	dt = 0.05
	# dt = 0.02
	# seed = args.seed
	# return_norm_th = True
	print("euler", euler)
	print("sigma",sigma)
	for file in os.listdir(root_path):
		if file.startswith(method+"_") and f"seed{tr_seed}" in file and f"samples{samples}" in file and f"sigma={sigma}" in file and f"euler={euler}" in file and file.endswith(".pth"):
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
	print(f"Bd for seed={tr_seed}", Bd)
	
	#make model for mpc
	model_type = 'discrete'
	model = do_mpc.model.Model(model_type)
	_phi = model.set_variable(var_type = '_x', var_name = 'phi', shape = (NKoopman,1))
	_u = model.set_variable(var_type = '_u', var_name = 'u', shape = (1,1))
	phi_next = Ad@_phi + Bd@_u
	model.set_rhs('phi',phi_next)
	phi_cost = _phi[0]**2 + 0.1 * _phi[1]**2  #should be (1,1) shape
	# phi_cost = _phi[1] **2 + 0.1 * _phi[2]**2
	u_cost = 0.001* _u**2
	model.set_expression(expr_name = 'phi_cost',expr = phi_cost)
	model.set_expression(expr_name = 'u_cost',expr = u_cost)
	model.setup()

	#make controller for mpc
	mpc = do_mpc.controller.MPC(model)
	setup_mpc = {
		'n_robust': 0,
		'n_horizon': 20,
		't_step': 0.05,
		'state_discretization':'discrete',
    	'store_full_solution':True,		
	}
	mpc.set_param(**setup_mpc)

	#make objective for mpc
	mterm = model.aux['phi_cost'] #terminal cost
	lterm = model.aux['u_cost'] + model.aux['phi_cost']# stage cost
	mpc.set_objective(mterm = mterm, lterm =lterm)

	# lower bounds of the input
	mpc.bounds['lower','_u','u'] = -2.

	# upper bounds of the input
	mpc.bounds['upper','_u','u'] =  2.

	#finish setup of mpc
	mpc.setup()

	#estimator/simulator
	estimator = do_mpc.estimator.StateFeedback(model)
	simulator = do_mpc.simulator.Simulator(model)
	simulator.set_param(t_step = 0.05)
	simulator.setup()


	# Kopt = dlqr(Ad,Bd,Q,R,gamma = gamma)


	#try evaluate over 100 init states
	# eval_seed = 100
	eval_seed = 0
	np.random.seed(eval_seed)
	max_steps = 200
	# n_init_states = 100
	# init_states = np.random.uniform(low = [-np.pi,-1.],
	# 	high = [np.pi, 1.], size = (n_init_states, 2))

	n_init_states = 1
	init_states = np.array([[np.pi,0.]])

	final_costs = np.empty(n_init_states)

	# env = noisyPendulumEnv(dt = dt)
	env = PendulumEnv(sigma = sigma, euler = euler)
	obs_dim = env.observation_space.shape[0]
	if save_traj == True:
		all_traj = np.empty((n_init_states,max_steps,obs_dim))

	for i in np.arange(n_init_states):
		init_state = init_states[i,:]
		# x0 = np.matrix(Psi_o(init_state,net,NKoopman)).reshape(NKoopman,1)
		x_ref_lift = Psi_o(x_ref,net,NKoopman).reshape(-1,1)
		eps_cost = 0
		obs = np.array(env.reset_state(state = init_state))
		# # mpc.x0 = x0 - x_ref_lift
		# obs = init_state
		for t in np.arange(max_steps):
			if save_traj == True:
				all_traj[i,t,:] = obs
			obs_lift = np.matrix(Psi_o(obs,net,NKoopman)).reshape(NKoopman,1)
			mpc.x0 = obs_lift - x_ref_lift
			u0 = mpc.make_step(obs_lift - x_ref_lift)
			u0 = np.array(u0[0,0])
			obs,reward,done,info  = env.step(u0)
			print("u0",u0)
			print("obs", obs)
			# print("model x0", model.x['phi'])
			# ucost = 0.001 * u0[0,0]**2
			# xcost = normalize(obs[0]) ** 2 + 0.1* obs[1,0] ** 2
			eps_cost += -reward


		print("eps cost for this trial", eps_cost)
		final_costs[i] = eps_cost

		# observations, u_list = exp(env, env_name, net,Kopt,init_state,x_ref,Nstate,
		# 	NKoopman,max_steps = max_steps, gamma = gamma)
		# print("u-list[:10]", u_list[:10])
		# cost_i = Cost(observations,u_list,Q,R,x_ref,gamma = gamma)
		# final_costs[i] = cost_i
		# print(f"final cost for init state {init_state}  is {cost_i}")
		# # vis_env = noisyPendulumEnv(dt = dt)
		# # vis_env.visualize(init_state = init_states[i], cmd = u_list)

	print(f"mean cost for tr_seed = {tr_seed}, sigma = {sigma}, euler={euler}", np.mean(final_costs))
	# print("final costs", final_costs)
	if save_traj == True:
		filename = f"traj_log/samples={samples}_koopman_tr_seed={tr_seed}_eval_seed={eval_seed}_sigma={sigma}_euler={euler}.npy"
		os.makedirs(os.path.dirname(filename), exist_ok = True)
		with open(filename, 'wb') as f:
			np.save(f, all_traj)
	final_costs_file = f"cost_log/samples={samples}_koopman_tr_seed={tr_seed}_eval_seed={eval_seed}_sigma={sigma}_euler={euler}.npy"
	os.makedirs(os.path.dirname(final_costs_file), exist_ok = True)
	with open(final_costs_file, 'wb') as f:
		np.save(f, final_costs)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--tr_seed", default=0, type=int) # specifies training seed to evalute
	parser.add_argument("--save_traj", default = "True")
	parser.add_argument("--samples", default = 80000) #tr samples with which model trained
	parser.add_argument("--sigma", type = float, default = 0.0)
	parser.add_argument("--euler", default = "False")
	args = parser.parse_args()
	save_traj = True if args.save_traj == "True" else False
	sigma = args.sigma
	euler = True if args.euler == "True" else False
	tr_seed = args.tr_seed
	samples = args.samples



	main(tr_seed = tr_seed, save_traj = save_traj, samples = samples, sigma = sigma, euler = euler)