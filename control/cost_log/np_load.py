import numpy as np
# mean_cost_arr = np.empty(4)
euler = False
sigmas = [0.0,1.0]
seeds = [0,1]
eval_seed = 100
for sigma in sigmas:
	sigma_cost = np.empty(len(seeds))
	for i in np.arange(len(seeds)):
		seed_costs = np.load(f"samples=80000_koopman_tr_seed={seeds[i]}_eval_seed={eval_seed}_sigma={sigma}_euler={euler}.npy")
		mean_cost = np.mean(seed_costs)
		sigma_cost[i] = mean_cost
	print(f" sigma = {sigma},  mean cost:", np.mean(sigma_cost))
	print(f" sigma = {sigma}, std:", np.std(sigma_cost))
		# mean_cost_arr[i] = mean_cost

