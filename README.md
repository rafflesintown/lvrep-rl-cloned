# CDC 2023 implementation

To replicate our implementation of Deep Koopman based control for the CDC 2023 paper, please use the Learn_Koopman_with_KlinearEig.py file in the train folder of this branch to train a new Koopman dynamics model. For evaluation, please use the og_eval_mpc.py file in the control folder of this branch. For general details of how to use the code in this branch, please refer to the following details from the original authors of this code: https://github.com/HaojieSHI98/DeepKoopmanWithControl.

# DeepKoopmanWithControl
Deep Koopman Operator with Control for Nonlinear Systems

Paper: https://arxiv.org/abs/2202.08004 (Accepted by RA-L 2022)

## Prediction
<img src="Prediction.png"  width="1000"/>
<img src="PredictionResults.png" width="1000">

## Control
<img src="Control.png" width = "1000">
<img src="FrankaStar.png" width = "1000">

## Requirement
``` python 
pytorch, gym, pybullet==3.2.1
```
## Environment
For gym environment, you should replace the gym env file with files in folder ./gym_env/

All the environments:
``` python 
"DampingPendulum"
"DoublePendulum"
"Franka"
"Pendulum-v1"
"MountainCarContinous-v0"
"CartPole-v1"
```

## Example
To train the network, you can just run 
``` python 
cd train/
python Learn_koopman_with_KlinearEig.py
```
To evaluate the prediction performance, you can utilize the notebooks in folder prediction/

To evaluate the control performance, you can utilize the notebooks in folder control/

For FrankaControl, please utilize the notebook in ./franka/evaluate_KoopmanAffline_Franka.ipynb and make sure pybullet==3.2.1
