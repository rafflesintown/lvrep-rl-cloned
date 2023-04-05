import gym
import numpy as np

class addNoisePendulum(gym.Wrapper):
	def __init__(self,env,sigma):
		super().__init__(env)
		# print("am I even here?")
		self.sigma = sigma
		# self.env.state_space = spaces.Box(low = -np.pi,high = np.pi,)

	# def step(self,action):
	# 	next_obs, reward, done, _ = self.env.step(action)
	# 	state_dim = self.env.state.shape
	# 	next_state = self.env.state + np.random.normal(size = state_dim,scale = self.sigma)
	# 	self.env.state = next_state
	# 	return next_state,reward,done,_
	
	def step(self,action):
		print("current state 1", self.env.state)
		next_obs, reward, done, _ = self.env.step(action)
		print("current state 2", self.env.state)
		state_dim = self.env.state.shape
		# self.env.state = np.random.normal(size = state_dim,scale = self.sigma)
		self.env.state = np.random.uniform(size = state_dim)
		# # next_obs = self.env._get_obs()
		next_obs = self.get_obs(self.env.state)
		# # print("next obs 2", next_obs)
		return next_obs,reward,done,_

	# def reset(self):
	# 	return self.env.reset()

	def get_obs(self,state): #only works for pendulum, obviously
		theta, thetadot = state
		return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

