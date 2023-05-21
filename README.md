# lvrep-rl modified by Zhaolin

To train an agent, use the main.py file (alternatively, the run_train.sh file). To evaluate an agent, use the eval.py file (alternatively, the run_eval.sh file).

The rfsac agent is the random fourier feature + SAC agent. The V critic network used by the rfsac agent is the RFVCritic critic in agent.rfsac.rfsac_agent file. The remaining elements of the agent (i.e. the actor) is an SAC actor.