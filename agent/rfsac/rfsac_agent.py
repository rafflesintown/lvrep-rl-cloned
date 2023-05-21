import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal 
import os

import torch.nn.init as init
import numpy as np
from utils import util 

# from utils.util import unpack_batch, RunningMeanStd
from utils.util import unpack_batch
from networks.policy import GaussianPolicy
from networks.vae import Encoder, Decoder, GaussianFeature
from agent.sac.sac_agent import SACAgent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
    """
    Critic with random fourier features
    """
    def __init__(
        self,
        feature_dim,
        num_noise=20, 
        hidden_dim=256,
        ):

        super().__init__()
        self.num_noise = num_noise
        self.noise = torch.randn(
            [self.num_noise, feature_dim], requires_grad=False, device=device)

        # Q1
        self.l1 = nn.Linear(feature_dim, hidden_dim) # random feature
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2
        self.l4 = nn.Linear(feature_dim, hidden_dim) # random feature
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)


    def forward(self, mean, log_std):
        """
        """
        std = log_std.exp()
        batch_size, d = mean.shape 
    
        x = mean[:, None, :] + std[:, None, :] * self.noise
        x = x.reshape(-1, d)

        q1 = F.elu(self.l1(x)) #F.relu(self.l1(x))
        q1 = q1.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
        q1 = F.elu(self.l2(q1)) #F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.elu(self.l4(x)) #F.relu(self.l4(x))
        q2 = q2.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
        q2 = F.elu(self.l5(q2)) #F.relu(self.l5(q2))
        # q2 = self.l3(q2) #is this wrong?
        q2 = self.l6(q2)

        return q1, q2




class RLNetwork(nn.Module):
    """
    An abstract class for neural networks in reinforcement learning (RL). In deep RL, many algorithms
    use DP algorithms. For example, DQN uses two neural networks: a main neural network and a target neural network.
    Parameters of a main neural network is periodically copied to a target neural network. This RLNetwork has a
    method called soft_update that implements this copying.
    """
    def __init__(self):
        super(RLNetwork, self).__init__()
        self.layers = []

    def forward(self, *x):
        return x

    def soft_update(self, target_nn: nn.Module, update_rate: float):
        """
        Update the parameters of the neural network by
            params1 = self.parameters()
            params2 = target_nn.parameters()

            for p1, p2 in zip(params1, params2):
                new_params = update_rate * p1.data + (1. - update_rate) * p2.data
                p1.data.copy_(new_params)

        :param target_nn:   DDPGActor used as explained above
        :param update_rate: update_rate used as explained above
        """

        params1 = self.parameters()
        params2 = target_nn.parameters()
        
        #bug? 
        for p1, p2 in zip(params1, params2):
            new_params = update_rate * p1.data + (1. - update_rate) * p2.data
            p1.data.copy_(new_params)

    def train(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()





#currently hardcoding sa_dim
#this is a  Q function
class RFQCritic(RLNetwork):
    def __init__(self, sa_dim = 4, embedding_dim = 4, n_neurons = 256):
        super().__init__()
        self.n_layers = 1
        self.n_neurons = n_neurons

        self.embed = nn.Linear(sa_dim, embedding_dim)

        # fourier_feats1 = nn.Linear(sa_dim, n_neurons)
        fourier_feats1 = nn.Linear(embedding_dim,n_neurons)
        init.normal_(fourier_feats1.weight)
        init.uniform_(fourier_feats1.bias, 0,2*np.pi)
        # init.zeros_(fourier_feats.bias)
        fourier_feats1.weight.requires_grad = False
        fourier_feats1.bias.requires_grad = False
        self.fourier1 = fourier_feats1 #unnormalized, no cosine/sine yet



        # fourier_feats2 = nn.Linear(sa_dim, n_neurons)
        fourier_feats2 = nn.Linear(embedding_dim,n_neurons)
        init.normal_(fourier_feats2.weight)
        init.uniform_(fourier_feats2.bias, 0,2*np.pi)
        fourier_feats2.weight.requires_grad = False
        fourier_feats2.bias.requires_grad = False
        self.fourier2 = fourier_feats2

        layer1 = nn.Linear( n_neurons, 1) #try default scaling
        # init.uniform_(layer1.weight, -3e-3,3e-3) #weight is the only thing we update
        init.zeros_(layer1.bias)
        layer1.bias.requires_grad = False #weight is the only thing we update
        self.output1 = layer1


        layer2 = nn.Linear( n_neurons, 1) #try default scaling
        # init.uniform_(layer2.weight, -3e-3,3e-3) 
        # init.uniform_(layer2.weight, -3e-4,3e-4)
        init.zeros_(layer2.bias)
        layer2.bias.requires_grad = False #weight is the only thing we update
        self.output2= layer2


    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        x = torch.cat([states,actions],axis = -1)
        # print("x initial norm", torch.linalg.norm(x))
        # x = F.batch_norm(x) #perform batch normalization (or is dbn better?)
        # x = (x - torch.mean(x, dim=0))/torch.std(x, dim=0) #normalization
        # x = self.bn(x)
        x = self.embed(x) #use an embedding layer
        # print("x norm after embedding", torch.linalg.norm(x))
        # print("layer1 norm", torch.linalg.norm(self.output1.weight))
        # x = F.relu(x)
        x1 = self.fourier1(x)
        x2 = self.fourier2(x)
        x1 = torch.cos(x1)
        x2 = torch.cos(x2)
        # x1 = torch.cos(x)
        # x2 = torch.sin(x)
        # x = torch.cat([x1,x2],axis = -1)
        # x = torch.div(x,1./np.sqrt(2 * self.n_neurons))
        x1 = torch.div(x1,1./np.sqrt(self.n_neurons)) #why was I multiplyigng?
        x2 = torch.div(x2,1./np.sqrt(self.n_neurons))
        # x1 = torch.div(x1,np.sqrt(self.n_neurons))
        # x2 = torch.div(x2,np.sqrt(self.n_neurons))
        # x1 = torch.div(x1,1./self.n_neurons)
        # x2 = torch.div(x2,1./self.n_neurons)
        # x = torch.relu(x)
        return self.output1(x1), self.output2(x2)





#currently hardcoding s_dim
#this is a  V function
class RFVCritic(RLNetwork):
    def __init__(self, s_dim = 3, embedding_dim = -1, rf_num = 256,sigma = 0.0, learn_rf = False):
        super().__init__()
        self.n_layers = 1
        self.n_neurons = rf_num

        self.sigma = sigma

        if embedding_dim != -1:
            self.embed = nn.Linear(s_dim, embedding_dim)
        else: #we don't add embed in this case
            embedding_dim = s_dim
            self.embed = nn.Linear(s_dim,s_dim)
            init.eye_(self.embed.weight)
            init.zeros_(self.embed.bias)
            self.embed.weight.requires_grad = False
            self.embed.bias.requires_grad = False

        # fourier_feats1 = nn.Linear(sa_dim, n_neurons)
        fourier_feats1 = nn.Linear(embedding_dim,self.n_neurons)
        # fourier_feats1 = nn.Linear(s_dim,n_neurons)
        if self.sigma > 0:
            init.normal_(fourier_feats1.weight, std = 1./self.sigma)
            # pass
        else:
            init.normal_(fourier_feats1.weight)
        init.uniform_(fourier_feats1.bias, 0,2*np.pi)
        # init.zeros_(fourier_feats.bias)
        fourier_feats1.weight.requires_grad = learn_rf
        fourier_feats1.bias.requires_grad = learn_rf
        self.fourier1 = fourier_feats1 #unnormalized, no cosine/sine yet



        fourier_feats2 = nn.Linear(embedding_dim, self.n_neurons)
        # fourier_feats2 = nn.Linear(s_dim,n_neurons)
        if self.sigma > 0:
            init.normal_(fourier_feats2.weight, std = 1./self.sigma)
            # pass
        else:
            init.normal_(fourier_feats2.weight)
        init.uniform_(fourier_feats2.bias, 0,2*np.pi)
        fourier_feats2.weight.requires_grad = learn_rf
        fourier_feats2.bias.requires_grad = learn_rf
        self.fourier2 = fourier_feats2

        layer1 = nn.Linear( self.n_neurons, 1) #try default scaling
        # init.uniform_(layer1.weight, -3e-3,3e-3) #weight is the only thing we update
        init.zeros_(layer1.bias)
        layer1.bias.requires_grad = False #weight is the only thing we update
        self.output1 = layer1


        layer2 = nn.Linear( self.n_neurons, 1) #try default scaling
        # init.uniform_(layer2.weight, -3e-3,3e-3) 
        # init.uniform_(layer2.weight, -3e-4,3e-4)
        init.zeros_(layer2.bias)
        layer2.bias.requires_grad = False #weight is the only thing we update
        self.output2= layer2


    def forward(self, states: torch.Tensor):
        x = states
        # print("x initial norm",torch.linalg.norm(x))
        # x = torch.cat([states,actions],axis = -1)
        # x = F.batch_norm(x) #perform batch normalization (or is dbn better?)
        # x = (x - torch.mean(x, dim=0))/torch.std(x, dim=0) #normalization
        # x = self.bn(x)
        x = self.embed(x) #use an embedding layer
        # print("x embedding norm", torch.linalg.norm(x))
        # x = F.relu(x)
        x1 = self.fourier1(x)
        x2 = self.fourier2(x)
        x1 = torch.cos(x1)
        x2 = torch.cos(x2)
        # x1 = torch.cos(x)
        # x2 = torch.sin(x)
        # x = torch.cat([x1,x2],axis = -1)
        # x = torch.div(x,1./np.sqrt(2 * self.n_neurons))
        # if self.sigma > 0:
        #   x1 = torch.multiply(x1,1./np.sqrt(2 * np.pi * self.sigma))
        #   x2 = torch.multiply(x2,1./np.sqrt(2 * np.pi * self.sigma)) 
        # x1 = torch.div(x1,np.sqrt(self.n_neurons/2))
        # x2 = torch.div(x2,np.sqrt(self.n_neurons/2))
        x1 = torch.div(x1,1./self.n_neurons)
        x2 = torch.div(x2,1./self.n_neurons)
        # print("x1 norm", torch.linalg.norm(x1,axis = 1))
        # x = torch.relu(x)
        return self.output1(x1), self.output2(x2)
        # return self.output1(x1),self.output1(x1) #just testing if the min actually helps

    def get_norm(self):
        l1_norm = torch.norm(self.output1)
        l2_norm = torch.norm(self.output2)
        return (l1_norm, l2_norm)




#currently hardcoding s_dim
#this is a  V function
#buffer: if not None, use samples from buffer to compute nystrom features
class nystromVCritic(RLNetwork):
    def __init__(self, s_dim = 3, s_low = np.array([-1,-1,-8]), feat_num = 256,sigma = 0.0, buffer = None):
        super().__init__()
        self.n_layers = 1
        self.n_neurons = feat_num

        self.sigma = sigma
        # if buffer is None:
        #     pass
        # else:
        #     self.init_using_samples(buffer, feat_num)

        s_high = -s_low

        #create nystrom feats 
        self.nystrom_samples1 = np.random.uniform(s_low,s_high,size = (feat_num, s_dim)) 
        self.nystrom_samples1 = torch.from_numpy(self.nystrom_samples1)
        self.nystrom_samples2 = np.random.uniform(s_low,s_high,size = (feat_num, s_dim)) 
        self.nystrom_samples2 = torch.from_numpy(self.nystrom_samples2)
        # self.nystrom_samples2 = torch.random.uniform(s_low,s_high,size = (feat_num,s_dim))
        if sigma > 0.0:
            self.kernel = lambda z: torch.exp(-torch.linalg.norm(z)**2/(2.* sigma**2))
        else:
            self.kernel = lambda z: torch.exp(-torch.linalg.norm(z)**2/(2.))
        self.K_m1 = self.make_K(self.nystrom_samples1,self.kernel)
        # self.K_m2 = self.make_K(self.nystrom_samples2,self.kernel)
        [eig_vals1,S1] = torch.linalg.eig(self.K_m1)
        # print("eig vals", eig_vals1)
        self.eig_vals1 = eig_vals1.float()
        self.S1 = S1.float()
        # [eig_vals2,S2] = torch.linalg.eig(self.K_m2)
        # self.eig_vals2 = eig_vals2.float()
        # self.S2 = S2.float()

        layer1 = nn.Linear( self.n_neurons, 1) #try default scaling
        # init.uniform_(layer1.weight, -3e-3,3e-3) #weight is the only thing we update
        init.zeros_(layer1.bias)
        layer1.bias.requires_grad = False #weight is the only thing we update
        self.output1 = layer1


        layer2 = nn.Linear( self.n_neurons, 1) #try default scaling
        # init.uniform_(layer2.weight, -3e-3,3e-3) 
        # init.uniform_(layer2.weight, -3e-4,3e-4)
        init.zeros_(layer2.bias)
        # print(layer2.weight.dtype, "weight dtype")
        layer2.bias.requires_grad = False #weight is the only thing we update
        self.output2= layer2

    # def init_using_samples(self, buffer, n_samples):
    #     #create nystrom feats 
    #     self.nystrom_samples1 = (buffer.sample(n_samples)).state
    #     print("nystrom samples", self.nystrom_samples1)
    #     if self.sigma > 0.0:
    #         self.kernel = lambda z: torch.exp(-torch.linalg.norm(z)**2/(2.* self.sigma**2))
    #     else:
    #         self.kernel = lambda z: torch.exp(-torch.linalg.norm(z)**2/(2.))
    #     self.K_m1 = self.make_K(self.nystrom_samples1,self.kernel)
    #     [eig_vals1,S1] = torch.linalg.eig(self.K_m1)
    #     print("eig vals", eig_vals1)
    #     self.eig_vals1 = eig_vals1.float()
    #     self.S1 = S1.float()  



    def make_K(self, samples,kernel):
        m,d = samples.shape
        K_m = torch.empty((m,m))
        for i in torch.arange(m):
            for j in torch.arange(m):
                K_m[i,j] = kernel(samples[i,:] - samples[j,:])
        return K_m


    def forward(self, states: torch.Tensor):
        # print("x initial norm",torch.linalg.norm(x))
        # x = torch.cat([states,actions],axis = -1)
        # x = F.batch_norm(x) #perform batch normalization (or is dbn better?)
        # x = (x - torch.mean(x, dim=0))/torch.std(x, dim=0) #normalization
        # x = self.bn(x)
        # x = self.embed(x) #use an embedding layer
        # print("x embedding norm", torch.linalg.norm(x))
        # x = F.relu(x)
        # print(states.shape, "shape1")
        # print("self sigma forward", self.sigma)
        x1 = self.nystrom_samples1.unsqueeze(0) - states.unsqueeze(1)
        # print(states.shape,"shape2")
        K_x1 = torch.exp(-torch.linalg.norm(x1,axis = 2)**2/2).float()
        # print("K_x1 nan?", torch.isnan(K_x1).any())
        # print("S1 nan", torch.isnan(self.S1).any())
        # print("eigvals nan", torch.isnan(self.eig_vals1**(-0.5)).any())
        # phi_all1 = K_x1 @ (self.S1).T
        # print(phi_all1.shape, "phi_all partial shape")
        phi_all1 = (K_x1 @ (self.S1)) @ torch.diag(self.eig_vals1**(-0.5))
        # print("phi_all norm", torch.linalg.norm(phi_all1, axis = 1))
        phi_all1 = phi_all1 * self.n_neurons
        # x2 = self.nystrom_samples2.unsqueeze(0) - states.unsqueeze(1)
        # phi_all2 = torch.exp(-torch.linalg.norm(x2,axis = 2)**2/2)
        # phi_all2 = phi_all2 * self.n_neurons * 3.
        # print("phi_all norm", torch.linalg.norm(phi_all1,axis=1))
        # phi_all = phi_all * np.sqrt(self.n_neurons) 
        # phi_all = torch.empty((len(states), self.n_neurons))
        # for i in range(len(states)):
        #     x = states[i,:]
        #     Kx = map(self.kernel, x - self.nystrom_samples)
        #     Kx =  torch.tensor(list(Kx)).float()
        #     phi_x = torch.diag(self.eig_vals**(-0.5)) @ ((self.S).T @Kx)
        #     phi_all[i,:] = phi_x
        # print("phi_all shape", phi_all.shape)
        phi_all1 = phi_all1.to(torch.float32)
        # phi_all2 = phi_all2.to(torch.float32)
        # print(phi_all.dtype, "phi all dtype")

        # x = torch.relu(x)
        # return self.output1(phi_all1), self.output2(phi_all2)
        return self.output1(phi_all1), self.output2(phi_all1)


    def get_norm(self):
        l1_norm = torch.norm(self.output1)
        l2_norm = torch.norm(self.output2)
        return (l1_norm, l2_norm)


# #currently hardcoding s_dim
# #really this is a V function rather than  Q function
# class RFCritic(RLNetwork):
#     def __init__(self, s_dim = 3, n_neurons = 256):
#         super().__init__()
#         self.n_layers = 1
#         self.n_neurons = n_neurons

#         fourier_feats = nn.Linear(s_dim, n_neurons)
#         init.normal_(fourier_feats.weight)
#         # init.uniform_(fourier_feats.bias, 0,2*np.pi)
#         init.zeros_(fourier_feats.bias)
#         fourier_feats.weight.requires_grad = False
#         fourier_feats.bias.requires_grad = False
#         self.fourier = fourier_feats #unnormalized, no cosine/sine yet

#         layer = nn.Linear( 2 * n_neurons, 1)
#         init.uniform_(layer.weight, -3e-3,3e-3) #weight is the only thing we update
#         init.zeros_(layer.bias)
#         layer.bias.requires_grad = False
#         self.output = layer
#     def forward(self, states: torch.Tensor):
#         x = states
#         x = self.fourier(x)
#         x1 = torch.cos(x)
#         x2 = torch.sin(x)
#         x = torch.cat([x1,x2],axis = -1)
#         x = torch.div(x,1./np.sqrt(2 * self.n_neurons))
#         # x = torch.div(x,1./self.n_neurons)
#         # x = torch.relu(x)
#         return self.output(x)


class DoubleQCritic(nn.Module):
  """Critic network, employes double Q-learning."""
  def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
    super().__init__()

    self.Q1 = util.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
    self.Q2 = util.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

    self.outputs = dict()
    self.apply(util.weight_init)

  def forward(self, obs, action):
    assert obs.size(0) == action.size(0)

    obs_action = torch.cat([obs, action], dim=-1)
    q1 = self.Q1(obs_action)
    q2 = self.Q2(obs_action)

    self.outputs['q1'] = q1
    self.outputs['q2'] = q2

    return q1, q2


class RFSACAgent(SACAgent):
    """
    SAC with random features
    """
    def __init__(
            self, 
            state_dim, 
            action_dim, 
            action_space, 
            # lr=1e-3,
            # lr = 5e-4,
            # lr = 3e-4,
            lr = 3e-4,
            discount=0.99, 
            target_update_period=2,
            tau=0.005,
            alpha=0.1,
            auto_entropy_tuning=True,
            hidden_dim=256,
            sigma = 0.0,
            rf_num = 256,
            learn_rf = False,
            use_nystrom = False,
            replay_buffer = None,
            # feature_tau=0.001,
            # feature_dim=256, # latent feature dim
            # use_feature_target=True, 
            # extra_feature_steps=1,
            ):

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            lr=lr,
            tau=tau,
            alpha=alpha,
            discount=discount,
            target_update_period=target_update_period,
            auto_entropy_tuning=auto_entropy_tuning,
            hidden_dim=hidden_dim,
        )

        # self.feature_dim = feature_dim
        # self.feature_tau = feature_tau
        # self.use_feature_target = use_feature_target
        # self.extra_feature_steps = extra_feature_steps

        # self.encoder = Encoder(state_dim=state_dim, 
        #       action_dim=action_dim, feature_dim=feature_dim).to(device)
        # self.decoder = Decoder(state_dim=state_dim,
        #       feature_dim=feature_dim).to(device)
        # self.f = GaussianFeature(state_dim=state_dim, 
        #       action_dim=action_dim, feature_dim=feature_dim).to(device)
        
        # if use_feature_target:
        #   self.f_target = copy.deepcopy(self.f)
        # self.feature_optimizer = torch.optim.Adam(
        #   list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.f.parameters()),
        #   lr=lr)

        # self.critic = RFCritic().to(device)
        # self.rfQcritic = RFQCritic().to(device)
        # self.rfQcritic_target = copy.deepcopy(self.rfQcritic)
        # self.critic_target = copy.deepcopy(self.critic)
        # self.critic_optimizer = torch.optim.Adam(
        #   self.critic.parameters(), lr=lr, betas=[0.9, 0.999])
        # self.rfQcritic_optimizer = torch.optim.Adam(
        #   self.rfQcritic.parameters(), lr=lr, betas=[0.9, 0.999])

        # self.critic = DoubleQCritic(
        #   obs_dim = state_dim,
        #   action_dim = action_dim,
        #   hidden_dim = hidden_dim,
        #   hidden_depth = 2,
        #   ).to(device)
        # self.critic = RFQCritic().to(device)
        if use_nystrom == False: #use RF
            self.critic = RFVCritic(sigma = sigma, rf_num = rf_num, learn_rf = learn_rf).to(device)
        else: #use nystrom
            feat_num = rf_num
            self.critic = nystromVCritic(sigma = sigma, feat_num = feat_num, buffer = replay_buffer).to(device)
        # self.critic = Critic().to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr, betas=[0.9, 0.999])





    # def feature_step(self, batch):
    #   """
    #   Feature learning step

    #   KL between two gaussian p1 and p2:

    #   log sigma_2 - log sigma_1 + sigma_1^2 (mu_1 - mu_2)^2 / 2 sigma_2^2 - 0.5
    #   """
    #   # ML loss
    #   z = self.encoder.sample(
    #       batch.state, batch.action, batch.next_state)
    #   x, r = self.decoder(z)
    #   s_loss = 0.5 * F.mse_loss(x, batch.next_state)
    #   r_loss = 0.5 * F.mse_loss(r, batch.reward)
    #   ml_loss = r_loss + s_loss

    #   # KL loss
    #   mean1, log_std1 = self.encoder(
    #       batch.state, batch.action, batch.next_state)
    #   mean2, log_std2 = self.f(batch.state, batch.action)
    #   var1 = (2 * log_std1).exp()
    #   var2 = (2 * log_std2).exp()
    #   kl_loss = log_std2 - log_std1 + 0.5 * (var1 + (mean1-mean2)**2) / var2 - 0.5
        
    #   loss = (ml_loss + kl_loss).mean()

    #   self.feature_optimizer.zero_grad()
    #   loss.backward()
    #   self.feature_optimizer.step()

    #   return {
    #       'vae_loss': loss.item(),
    #       'ml_loss': ml_loss.mean().item(),
    #       'kl_loss': kl_loss.mean().item(),
    #       's_loss': s_loss.mean().item(),
    #       'r_loss': r_loss.mean().item()
    #   }

    #inputs are tensors
    def get_reward(self, states,action):
        th = torch.atan2(states[:,1],states[:,0]) #1 is sin, 0 is cosine 
        thdot = states[:,2]
        action = torch.reshape(action, (action.shape[0],))
        # print("th shape", th.shape)
        # print("thdot shape", thdot.shape)
        # print('action shape', action.shape)
        th = self.angle_normalize(th)
        reward = -(th**2 + 0.1* thdot**2 + 0.01*action**2)
        return torch.reshape(reward,(reward.shape[0],1))

    def angle_normalize(self,th):
        return((th + np.pi) % (2 * np.pi)) -np.pi
    def f_star_2d(self,states,action,g = 10.0,m = 1.,l=1.,max_a = 2.,max_speed = 8.,dt = 0.05):
        th = torch.atan2(states[:,1],states[:,0]) #1 is sin, 0 is cosine 
        thdot = states[:,2]
        action = torch.reshape(action, (action.shape[0],))
        u = torch.clip(action,-max_a,max_a)
        newthdot = thdot +(3. * g / (2 * l) * torch.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = torch.clip(newthdot, -max_speed,max_speed)
        newth = th + newthdot * dt
        # new_states = torch.empty((states.shape[0],3))
        # # print("new states shape 1", new_states.shape)
        # new_states[:,0] = torch.cos(newth)
        # new_states[:,1] = torch.sin(newth)
        # new_states[:,2] = newthdot
        # print("new states shape", new_states.shape)
        new_states = torch.empty((states.shape[0],2))
        new_states[:,0] = self.angle_normalize(newth)
        new_states[:,1] = newthdot
        return new_states

    #this returns cos(th), sin(th), thdot
    def f_star_3d(self,states,action,g = 10.0,m = 1.,l=1.,max_a = 2.,max_speed = 8.,dt = 0.05):
        th = torch.atan2(states[:,1],states[:,0]) #1 is sin, 0 is cosine 
        thdot = states[:,2]
        action = torch.reshape(action, (action.shape[0],))
        u = torch.clip(action,-max_a,max_a)
        newthdot = thdot +(3. * g / (2 * l) * torch.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = torch.clip(newthdot, -max_speed,max_speed)
        newth = th + newthdot * dt
        # new_states = torch.empty((states.shape[0],3))
        # # print("new states shape 1", new_states.shape)
        # new_states[:,0] = torch.cos(newth)
        # new_states[:,1] = torch.sin(newth)
        # new_states[:,2] = newthdot
        # print("new states shape", new_states.shape)
        new_states = torch.empty((states.shape[0],3))
        new_states[:,0] = torch.cos(newth)
        new_states[:,1] = torch.sin(newth)
        new_states[:,2] = newthdot
        return new_states       

    def update_actor_and_alpha(self, batch):
        """
        Actor update step
        """
        # dist = self.actor(batch.state, batch.next_state)
        dist = self.actor(batch.state)
        action = dist.rsample()
        # print("action shape", action.shape)
        # print("batch state shape", batch.state.shape)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        # if self.use_feature_target:
        #   mean, log_std = self.f_target(batch.state, action)
        # else:
        #   mean, log_std = self.f(batch.state, action)
        # q1, q2 = self.critic(mean, log_std)
        # q = torch.min(q1, q2)
        # q = self.discount * self.critic(batch.next_state) + batch.reward 
        # q = batch.reward 
        # q1,q2 = self.rfQcritic(batch.state,batch.action)
        # q1,q2 = self.critic(batch.state,action) #not batch.action!!!
        # q = torch.min(q1, q2)
        # q = q1 #try not using q1, q1
        reward = self.get_reward(batch.state,action) #use reward in q-fn
        # print("reward shape", reward.shape)
        # q1,q2 = self.critic(batch.state,action)
        # q1, q2 = self.critic(self.f_star(batch.state,action))
        q1,q2 = self.critic(self.f_star_3d(batch.state,action))
        # print("q1 shape",q1.shape)
        # q = self.discount * torch.min(q1,q2) + reward
        q = self.discount * torch.min(q1,q2) + reward

        actor_loss = ((self.alpha) * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        info = {'actor_loss': actor_loss.item()}

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                                        (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            info['alpha_loss'] = alpha_loss 
            info['alpha'] = self.alpha 

        return info


    def critic_step(self, batch):
        """
        Critic update step
        """         
        # state, action, reward, next_state, next_action, next_reward,next_next_state, done = unpack_batch(batch)
        state,action,next_state,reward,done = unpack_batch(batch)
        
        with torch.no_grad():
            dist = self.actor(next_state)
            next_action = dist.rsample()
            next_action_log_pi = dist.log_prob(next_action).sum(-1, keepdim=True)

            # if self.use_feature_target:
            #   mean, log_std = self.f_target(state, action)
            #   next_mean, next_log_std = self.f_target(next_state, next_action)
            # else:
            #   mean, log_std = self.f(state, action)
            #   next_mean, next_log_std = self.f(next_state, next_action)


            # next_q1, next_q2 = self.critic_target(next_state, next_action)
            # next_q1, next_q2 = self.critic_target(self.f_star(next_state,next_action))
            next_q1, next_q2 = self.critic_target(self.f_star_3d(next_state,next_action))
            next_q = torch.min(next_q1,next_q2)-  self.alpha * next_action_log_pi
            next_reward = self.get_reward(next_state,next_action) #reward for new s,a
            # target_q = reward + (1. - done) * self.discount * next_q 
            # target_q = next_reward + (1. - done) * self.discount * next_q 
            target_q = next_reward + (1. - done) * self.discount * next_q

            
        # q1, q2 = self.critic(mean, log_std)
        # q1,q2 = self.rfQcritic(state,action)
        # q1,q2 = self.critic(state,action)
        # q1,q2 = self.critic(self.f_star(state,action))
        q1,q2 = self.critic(self.f_star_3d(state,action))
        q1_loss = F.mse_loss(target_q, q1)
        q2_loss = F.mse_loss(target_q, q2)
        q_loss = q1_loss + q2_loss
        # q = self.critic(next_state)
        # q = self.rfQcritic(state,action)
        # q_loss = F.mse_loss(target_q,q)
        # q1,q2 = self.rfQcritic(state,action)
        # q = torch.min(q1,q2)
        # q_loss = F.mse_loss(target_q,q)

        self.critic_optimizer.zero_grad()
        # self.rfQcritic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()
        # self.rfQcritic_optimizer.step()

        return {
            'q1_loss': q1_loss.item(), 
            'q2_loss': q2_loss.item(),
            'q1': q1.mean().item(),
            'q2': q2.mean().item()
            }


    def update_feature_target(self):
        for param, target_param in zip(self.f.parameters(), self.f_target.parameters()):
            target_param.data.copy_(self.feature_tau * param.data + (1 - self.feature_tau) * target_param.data)
    

    def train(self, buffer, batch_size):
        """
        One train step
        """
        self.steps += 1

        # # Feature step
        # for _ in range(self.extra_feature_steps+1):
        #   batch = buffer.sample(batch_size)
        #   feature_info = self.feature_step(batch)

        #   # Update the feature network if needed
        #   if self.use_feature_target:
        #       self.update_feature_target()

        batch = buffer.sample(batch_size)

        # Acritic step
        critic_info = self.critic_step(batch)
        # critic_info = self.rfQcritic_step(batch)

        # Actor and alpha step
        actor_info = self.update_actor_and_alpha(batch)

        # Update the frozen target models
        self.update_target()

        return {
            # **feature_info,
            **critic_info, 
            **actor_info,
        }


    
