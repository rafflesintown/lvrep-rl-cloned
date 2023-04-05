import numpy as np
# mean_cost_arr = np.empty(4)
euler = False
sigmas = [0.0,1.0]
seeds = [0,1,2,3]


for sigma in sigmas:
	if sigma == 0: 
		costs = np.load("seed=0_euler=%r_sigma=0.0.npy"%(euler))
		mean_cost = np.mean(costs)
		print("mean for no noise", mean_cost)
	else:
		all_cost = np.empty(len(seeds))
		for i in seeds:
			seed_costs = np.load("seed=%d_euler=%r_sigma=%.1f.npy"%(i,euler,sigma))
			seed_mean = np.mean(seed_costs)
			all_cost[i] = seed_mean
		print("mean for sigma %.1f" %sigma, np.mean(all_cost))
		print("std for sigma %.1f" % sigma, np.std(all_cost))



# for i in seeds:
# 	if i == 0:
# 		for sigma in sigmas:
# 			seed_costs = np.load("seed=%d_euler=%r_sigma=%.1f.npy"%(i,euler,sigma))
# 			mean_cost = np.mean(seed_costs)
# 			print(f"tr seed {i} cost, euler={euler}, sigma = {sigma} ", mean_cost)
# 			# mean_cost_arr[i] = mean_cost
# 	else:
# 		for sigma in [0.5,1]:
# 			seed_costs = np.load("seed=%d_euler=%r_sigma=%.1f.npy"%(i,euler,sigma))
# 			mean_cost = np.mean(seed_costs)
# 			print(f"tr seed {i} cost, euler={euler}, sigma = {sigma} ", mean_cost)		

# print("mean cost over 4 training seeds: ", np.mean(mean_cost_arr))
# print("stdev over 4 training seeds: ", np.std(mean_cost_arr))