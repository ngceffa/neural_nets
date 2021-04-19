import numpy as np
import nn_utility_functions as nn_utils
import importlib
importlib.reload(nn_utils)
import matplotlib.pyplot as plt
from Connectivity_3 import Connectivity
import os
import torch
import time


class MBmodel(Connectivity):

    def __init__(self,
                 path,
                 valence=[-1, 1],
                 batches=30,
                 input_size=70, # Number of Kenyon cells
                 input_fraction=.1, # KC coding level
                 r_size=1, #
                 total_time=80,
                 dt=.5,
                 output_size=1,
                 reset_times=[30, 60],
                 cutoff_hemisphere_connections=2,
                 cutoff_homologous_connections=9,
                 method='ashock_2020', # or 'full'
                 condition='control',
                 epochs=1,
                 mbon_initial_rates=0,
                 mbin_initial_rates=0.1,
                 fb_ff_initial_rates=0.1
                 ):
        super().__init__(path, cutoff_hemisphere_connections)
        us_size = len(valence)
        self.mbon_initial_rates = mbon_initial_rates
        self.mbin_initial_rates = mbin_initial_rates
        self.fb_ff_initial_rates = fb_ff_initial_rates
        self.reset_times = [int(reset_times[0] / dt),
                            int(reset_times[1] / dt)]
        self.input_size = input_size
        self.input_fraction = input_fraction
        self.batches = batches
        self.output_size=output_size
        self.dt = dt
        self.timesteps = int(total_time / dt)
        self.s0 = np.zeros([self.timesteps,
                             input_size,
                             batches],
                             dtype=np.float32) # KC inputs (time, cell_index)
        self.wmax = 1 / (input_size * input_fraction)
                                            # Maximum KC-to-MBON synaptic weight
        self.w_init = .5 * self.wmax

        self.weights_KC = torch.zeros([self.matrix.shape[0],
                                       input_size,
                                       batches], dtype=torch.float32)
        # self.weights_KC[self.neuron_idxs["MBON"],:,:] = self.w_init
        self.n_neurons = self.matrix.shape[0]
        self.neuron_types = ["MBON", "all_mbins", "fb_ff"]
        self.rates = torch.zeros(size=(self.n_neurons, self.batches))
        self.learning_rate = 2 * np.logspace(3, -3, epochs)
        self.opt = torch.optim.RMSprop(self.train_vars, lr=self.learning_rate[0])

        self.DAN_MBON_indices = torch.ones(size=self.matrix.shape)
        for pre in self.indices["all_mbins"]:
            for post in self.indices["MBON"]:
                self.DAN_MBON_indices[post, pre] = 0

    def reset_initial_conditions(self,
                               mbon_initial_rates=0,
                               mbin_initial_rates=0.1,
                               fb_ff_initial_rates=0.1,
                               tau=5, # Timing-dependent plasticity window
                               tauw=5 # Timescale of weight modifications
                               ):
        # create a bunch of variables for the simulation
        self.mbon_rates_0 = mbon_initial_rates \
                            * np.ones([self.mbons_number, 
                            self.batches], dtype=np.float32)
        self.mbin_rates_0 = mbin_initial_rates \
                            * np.ones([self.mbins_number, 
                            self.batches], dtype=np.float32)
        self.fb_ff_rates_0 = fb_ff_initial_rates \
                            * np.ones([self.fb_ff_number, 
                            self.batches], dtype=np.float32)
        self.w0 = np.ones([self.mbons_number, # W (KC->MBON)
                      self.input_size,
                      self.batches],
                      dtype=np.float32) / (self.input_size * self.input_fraction)
        self.sbar0 = np.zeros([self.input_size, self.batches],
                               dtype=np.float32)
        self.dabar0 = np.zeros([self.mbons_number, self.batches],
                               dtype=np.float32)
        self.ma = np.zeros([self.timesteps, self.mbons_number, self.batches])
        self.da = np.zeros([self.timesteps, self.mbins_number, self.batches])
        self.oa = np.zeros([self.timesteps, self.fb_ff_number, self.batches])
        self.wa = np.zeros([self.timesteps,
                            self.mbons_number,
                            self.input_size,
                            self.batches])
        self.r = torch.zeros(self.timesteps, self.output_size, self.batches)
        # output_size is always 1...
        self.dacost = torch.zeros(1)
        for ntype, init_val in zip(self.neuron_types, [mbon_initial_rates,
                                                       mbin_initial_rates,
                                                       fb_ff_initial_rates]):
            self.rates[self.indices[ntype], :] = init_val
        self.lowpass_MBONdopamine = torch.zeros([self.mbons_number,
                                                 self.batches])
        self.lowpass_KC_rates = torch.zeros([self.input_size, self.batches])

        self.weights_KC = torch.zeros(self.n_neurons,
                                      self.input_size,
                                      self.batches)
        self.weights_KC[self.indices["MBON"], :, :] = self.w_init

    def train_first_order(self,
              task='first_order',
              epochs=1,
              show=False):
        # Variables common for every epoch
        self.loss = np.zeros((self.batches, epochs))
        self.learning_rate = 2 * np.logspace(3, -3, epochs)
        alphada = .1 # ?
        modelfname = os.getcwd() + "first_order.pt"
        opt = torch.optim.RMSprop(self.train_vars, lr=self.learning_rate[0])
        # check documentation of this opt
        self.track_loss = np.zeros(epochs)
        self.track_loss_da = np.zeros(epochs)
        odors, valences = nn_utils.odor_valence_pairs(num_cells=70, num_odors=2)
        
        start = time.time()

        for epoch in range(epochs):
            for g in opt.param_groups:
                g['lr'] = self.learning_rate[epoch]

            cs, us, rtarg = nn_utils.first_order_xp_2(odors, 
                                                      valences,
                                                      num_trials=self.batches)
            baselineinds = (np.sum(cs,1,keepdims=True) == 0) \
                     & (np.sum(us,1,keepdims=True) == 0)
            baselineinds_d = torch.from_numpy(
                        np.repeat(
                        baselineinds,
                        self.mbins_number, axis=1).astype(np.float32))
            # cs = KCs that are ON, time X KC x batch
            # us = US, time x batch
            # rtarg = target output (expected valence), time x batch

            m = torch.from_numpy(self.mbon_rates_0) # mbons variable rates
            d = torch.from_numpy(self.mbin_rates_0) # mbins variable rates
            o = torch.from_numpy(self.fb_ff_rates_0) # feedback/feedforward
                                                     # variable rates
            w = torch.from_numpy(self.w0) # W (KC->MBON),
                                          # used for an update rule input
            wfast = torch.from_numpy(self.w0)

            ma = np.zeros([self.timesteps, self.mbons_number, self.batches])
            da = np.zeros((self.timesteps, self.mbins_number, self.batches))
            oa = np.zeros((self.timesteps, self.fb_ff_number, self.batches))
            wa = np.zeros((self.timesteps,
                           self.mbons_number,
                           self.input_size, # number of KCs
                           self.batches))
            r = torch.zeros(self.timesteps, 1, self.batches)
            dacost = torch.zeros(1)

            sbar = torch.from_numpy(self.sbar0)
            dastdpbar = torch.from_numpy(self.dabar0) 

            # actual work is done in here:
            for ti in range(self.timesteps):
                mnew = ((1 - self.dt) * m
                        + self.dt
                        * torch.relu(torch.tanh(
                          self.J['mm'].mm(m)
                        + self.J['mf'].mm(o) # missing one? J['md']
                        + torch.einsum('ijb,jb->ib', 
                                        (w.float(), cs[ti,:,:].float()))
                        + self.bm)))

                dnew = ((1 - self.dt) * d
                        + self.dt
                        * torch.relu(torch.tanh(
                          self.J['dm'].mm(m)
                        + self.J['dd'].mm(d)
                        + self.J['df'].mm(o)
                        + self.w_mbins.float().mm(us[ti, :, :].float()) 
                        # ext input from US
                        + self.bd)))

                onew = ((1 - self.dt) * o 
                        + self.dt
                        * torch.relu(torch.tanh(
                            self.J['fm'].mm(m)
                        + self.J['fd'].mm(d)
                        + self.J['ff'].mm(o)
                        + self.w_fb_ff.float().mm(us[ti, :, :].float()) 
                        # ext input from US
                        + self.bf)))
                m = mnew
                d = dnew
                o = onew
                r[ti, :, :] = self.wrm.mm(m) # value readout (reward / punish)
                
                # "dopamine" level in MBONS, x batch
                mbin_to_mbon = torch.relu(self.J['md'].mm(d))
                # expanded on KCs

                stdp_update = -torch.einsum('ib,jb->ijb', (mbin_to_mbon, sbar)) \
                    + torch.einsum('ib,jb->ijb',(dastdpbar, cs[ti,:,:])) 
                    # anti-Hebbian --> search this...
                
                wfast = torch.relu(w + dt*(stdp_update \
                    - torch.relu(stdp_update - (self.wmax - w))))
                    #update that does not exceed wmax
                w = w + (dt/tauw)*(wfast - w) # eq 3
                
                dastdpbar = (1. - dt/tau)*dastdpbar + (dt/tau)*dastdp
                sbar = (1. - dt/tau)*sbar + (dt/tau)*cs[ti,:,:]

                dacost += torch.sum(torch.pow(
                          torch.relu(
                              d * baselineinds_d[ti,:,:]
                              - self.mbon_initial_rates), 2))
            # epochs cycle
            loss_err = torch.sum(torch.pow(r - rtarg,2)) / self.batches
            loss_da = alphada*dacost / self.batches
            loss = loss_err + loss_da
            track_loss[epoch] = loss_err
            track_loss_da[epoch] = loss_da
            
            loss.backward()
            opt.step()
            opt.zero_grad()


            # there must be a better way to select the ones
            #constraints
            #sparse connectivity -> keep zeros at zero
            self.J['mm'].data = self.J['mm'].data * (self.W['mm0'] != 0)
            Jmd.data = Jmd.data * Jmdbin
            Jmo.data = Jmo.data * Jmobin
            Jdm.data = Jdm.data * Jdmbin
            Jdd.data = Jdd.data * Jddbin
            Jdo.data = Jdo.data * Jdobin
            Jom.data = Jom.data * Jombin
            Jod.data = Jod.data * Jodbin
            Joo.data = Joo.data * Joobin
            #excitatory/inhibitory neurons
            Jmm.data[:,mexc] = torch.relu(Jmm.data[:,mexc])
            Jdm.data[:,mexc] = torch.relu(Jdm.data[:,mexc])
            Jom.data[:,mexc] = torch.relu(Jom.data[:,mexc])
            Jmo.data[:,oexc] = torch.relu(Jmo.data[:,oexc])
            Jdo.data[:,oexc] = torch.relu(Jdo.data[:,oexc])
            Joo.data[:,oexc] = torch.relu(Joo.data[:,oexc])
            Jmm.data[:,minh] = -torch.relu(-Jmm.data[:,minh])
            Jdm.data[:,minh] = -torch.relu(-Jdm.data[:,minh])
            Jom.data[:,minh] = -torch.relu(-Jom.data[:,minh])
            Jmo.data[:,oinh] = -torch.relu(-Jmo.data[:,oinh])
            Jdo.data[:,oinh] = -torch.relu(-Jdo.data[:,oinh])
            Joo.data[:,oinh] = -torch.relu(-Joo.data[:,oinh])
            
            #MBON to PI mapping HEREEEEEEEEEE
            wrm.data[:,mpos] = torch.relu(wrm.data[:,mpos])
            wrm.data[:,mneg] = -torch.relu(-wrm.data[:,mneg])
    


if __name__=="__main__":
    pass

    plt.style.use('grayscale')

    file_path = os.getcwd() + '/eschbach_2020/data'

    model = MBmodel(file_path)
    model.reset_initial_conditions()
    # model.setNetworkParameters(model.train_vars)
    # already def in Connectivity_3 --> __init__

    # these 2 are also attributes:
    epochs = 1
    learnig_rate = 2 * np.logspace(3, -3, epochs)

    odors, valences = nn_utils.odor_valence_pairs(num_cells=70, num_odors=2)
    # n.b. assume all-to-all KCs-MBONs connections


















        # run the equationsn 1 --> 3
        # cs, us = self.KC_rate, self.US_rate
        
        #tens.close()
        # w, b, w_us = [copyTensor(tens) for tens in [
        # self.weights, self.biases, self.us_weights]] 
        # 
        # # copy variables that are due to be trained

        # r, w_kc = self.rates, self.weights_KC

        # # Set DAN-MBON weights to zero
        # w *= self.DAN_MBON_indices

        # # Define inputs
        # I_internal = torch.mm(w,r) + b #  (w.mm(r) + b).clone() 
        # # (weights * rates) + biases
        # I_KC = torch.einsum('ijb,jb->ib',(w_kc, cs)) # KC inputs
        # I_US = torch.mm(w_us,us) #w_us.mm_(us) # US inputs

        # # Define non-linear, positive rectification function
        # nonlinear = lambda x : torch.relu(torch.tanh(x))
        
        # # Return new firing rate (i.e. Equation 1 in original paper)
        # self.rates = (1-dt)*r + dt*nonlinear(I_internal + I_KC + I_US)

    


    # a = torch.einsum('ijb,jb->ib',(model.w, model.s[0, :, :]))