import numpy as np

all_traj = np.load(f"samples=80000_koopman_tr_seed=0_eval_seed=0.npy")
print("first 10 init states", all_traj[:10,0,:])

