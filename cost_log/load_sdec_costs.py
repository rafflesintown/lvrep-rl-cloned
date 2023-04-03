import numpy as np
# mean_cost_arr = np.empty(4)
euler = False
sigmas = [0.0,1.0]
seeds = [0,1,2,3]
eval_seed = 0
for sigma in sigmas:
	mean_cost = np.empty(len(seeds))
	std_cost = np.empty(len(seeds))
	for i in np.arange(len(seeds)):
		mean_cost[i] = np.mean(np.load(f"cost_rf_num=512_learn_rf=False_tr_seed={seeds[i]}_eval_seed={eval_seed}_euler={euler}_sigma={sigma}.npy"))
		std_cost[i] = np.std(np.load(f"cost_rf_num=512_learn_rf=False_tr_seed={seeds[i]}_eval_seed={eval_seed}_euler={euler}_sigma={sigma}.npy"))

	print("mean cost for sigma = %.1f" %sigma,  np.mean(mean_cost))
	print("std cost for sigma = %.1f" %sigma,  np.std(mean_cost))
	

# print("mean cost over 4 training seeds: ", np.mean(mean_cost_arr))
# print("stdev over 4 training seeds: ", np.std(mean_cost_arr))