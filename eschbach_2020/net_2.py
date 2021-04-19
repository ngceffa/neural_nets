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
    
    # --------------------------------------------------------------------------
    # almost untouched functions from BaserNetwork.py: 
    def setNetworkParameters(self, train_vars):
        self.weights, self.biases, self.us_weights, self.MBONbiases = train_vars
    
    def setPrePostWeights(self, val, pres, posts):
        for post in posts:
            for pre in pres:
                self.weights[post, pre] = val
    
    def load(self, filename):
        '''Filename = Torch save file (i.e. .p / .pt)'''
        # Load variables
        loadedVars = torch.load(filename)
        if len(loadedVars) == 2:
            train_vars, self.loss = loadedVars
        elif len(loadedVars) == 3:
            train_vars, self.parameters, self.loss = loadedVars
            self.setNetworkParameters(train_vars)
        elif len(loadedVars) == 4:
            train_vars, self.parameters, self.loss, self.taskScripts = loadedVars
            self.setNetworkParameters(train_vars)
        elif len(loadedVars) == 5:
            train_vars, self.parameters, self.loss, \
                self.bestModel, self.taskScripts = loadedVars
            self.setNetworkParameters(train_vars)

    # this is self.indices['MBON'] or ['all_mbins] or ['fb_ff']
    #self.neuron_idxs[ntype] = getNeuronIndices(nnames) 
    # # Get indices of each neuron type
    # this is self.mbons_number, self.mbins_number, self.fb_ff_number
    #self.neuron_numbers[ntype] = len(nnames)

    def initialiseRecordings(self):
        self.totalTime = self.timesteps # self.KC_rates.shape[0] ?
        self.recordings['rates'] = torch.zeros((self.timesteps,
                                                n_neurons,
                                                self.batches))
        self.recordings['output_valence'] = torch.zeros((self.timesteps,
                                                         1,
                                                         self.batches))
        self.recordings['MBONdopamine'] = torch.zeros((self.timesteps,
                                                      self.mbons_number,
                                                      self.batches))
        self.recordings['lowpass_MBONdopamine'] = torch.zeros((self.timesteps,
                                                               self.mbons_number,
                                                               self.batches))
        self.recordings['lowpass_KC_rates'] = torch.zeros(self.timesteps,
                                                          self.input_size,
                                                          self.batches)
        self.recordings['weight_updates'] = torch.zeros(self.timesteps,
                                                        self.mbons_number,
                                                        self.input_size,
                                                        self.batches)
        self.recordings['weights_KC'] = torch.zeros(self.timesteps,
                                                    self.mbons_number,
                                                    self.input_size, 
                                                    self.batches)

    def resetTaskValues(self, cs, us, target): # reset instead of set
        self.KC_rates = torch.Tensor(cs)
        self.US_rates = torch.Tensor(us)
        self.target_output = torch.Tensor(target)
        self.initialiseRecordings()
    
    def updateTaskValues(self, func, parameters, taskScript):
        self.taskGenerator = func
        self.taskParameters = parameters
        self.taskScript = taskScript
        rDict = self.taskGenerator() 
        # get variables returned by task generator function
        self.recordings["KC_rates"] = rDict["KC_rates"] # save KC rates
        for key in ["KC_rates", "US_rates", "target_output"]:
            if key not in rDict.keys():
                raise Exception(f'{key} missing from task input variables')
        for var in rDict.keys():
            exec(f"self.{var} = torch.Tensor(rDict['{var}'])")
        self.initialiseRecordings()

    def removeNeurons(self): # lesioning
        '''Remove neurons by setting their activities to zero.
        NOTE: This function must be called on every iteration t of self.run()'''
        self.rates[self.neuronsToRemove,:] = 0
    # --------------------------------------------------------------------------

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
        self.sbar0 = np.zeros([self.input_size, self.batches], dtype=np.float32)
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
            self.reset_initial_conditions()

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
                if ti in self.reset_times:
                    mnew = torch.from_numpy(self.mbon_rates_0)
                    dnew = torch.from_numpy(self.mbin_rates_0)
                    onew = torch.from_numpy(self.fb_ff_rates_0)
                    sbar = torch.from_numpy(self.sbar0)
                    # dastbar0 and badar0 are the same thing
                    dastdpbar = torch.from_numpy(self.dabar0) 
                else:
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
                ma[ti, :, :] = m.detach()
                da[ti, :, :] = d.detach()
                oa[ti, :, :] = o.detach()
                wa[ti, :, :, :] = w.detach()
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
    model.train_first_order()

    # for epoch in range(epochs):

    #     # reset initial conditions
    #     model.reset_initial_conditions()

    #     # cond stim; uncond stim; target response (known valence)
    #     # --> "set task values"
    #     cs, us, target = nn_utils.first_order_xp_2(odors, valences)

    #     for t in range(model.timesteps):
    #         # quantities _tbt = to-be-trained

    #         cs_tbt, us_tbt = cs[t, : , :], us[t, : , :]

    #         weights_tbt = torch.clone(model.weights)
    #         biases_tbt = torch.clone(model.biases)
    #         weights_us_tbt = torch.clone(model.us_weights)

    #         rates_tbt, w_kc_tbt = model.rates, model.weights_KC

    #         weights_tbt *= model.DAN_MBON_indices # put to 0 mbins-->mbon
    #         # MBIN only can reinforce KC-MBON but not touch MBON directly...

    #         # Define inputs
    #         I_internal = torch.mm(weights_tbt, rates_tbt) + biases_tbt
    #         # KC inputs
    #         I_KC = torch.einsum('ijb,jb->ib',(w_kc_tbt.float(), cs_tbt.float())) 
    #         I_US = torch.mm(weights_us_tbt.float(), us_tbt.float())
    #         # Define non-linear, positive rectification function
    #         nonlinear = lambda x : torch.relu(torch.tanh(x))
    #         # Return new firing rate (i.e. Equation 1 in original paper)
    #         model.rates = (1 - model.dt) * rates_tbt \
    #                    + model.dt * nn_utils.fire_func(I_internal + I_KC + I_US)
            
    #         # Get DAN firing rates, and MBON dopamine levels
    #         DAN_rates = model.rates[model.indices["all_mbins"], :]

    #         DAN_MBON_weights = model.weights(model.indices["all_mbins"], 
    #                                         model.indices["MBON"]).clone()
    #         # DAN firing rates * DAN-MBON weights
    #         MBONdopamine = torch.relu(DAN_MBON_weights.mm(DAN_rates)) 
    #         # KC-MBON_weights
    #         weights_tbt = modeel.weights_KC[model.indices["MBON"],:]

    #         # Multiply MBON dopamine with KC firing rates
    #         stdp_update = - torch.einsum('ib,jb->ijb', (
    #                                 MBONdopamine, model.lowpass_KC_rates))\
    #                       + torch.einsum('ib,jb->ijb', (
    #                                 model.lowpass_MBONdopamine, model.KC_rate))
    #         #update that does not exceed wmax
    #         next_w = torch.relu(weights_tbt \
    #                             + model.dt * (
    #                             stdp_update - torch.relu(
    #                             stdp_update - (wmax - w))))
    #         # current change is equal 
    #         # to current difference between previous (weights_tbt) and (next_w)
    #         weight_updates = (model.dt / model.tauw)\
    #                         * (next_w - weights_tbt) 
                            
    #         weights_tbt += weight_updates
    #         model.weights_KC[model.indices["MBON"],:,:] = weights_tbt

    #         # Update low-pass filter variables
    #         lowpass = lambda low, orig : (1.-dt/tau)*low + (dt/tau)*orig
    #         model.lowpass_MBONdopamine = \
    #                     1 - model.dt / model.tau * model.lowpass_MBONdopamine\
    #                     + model.dt / model.tau * model.MBONdopamine)
    #         model.lowpass_KC_rates = \
    #             1 - model.dt / model.tau * model.lowpass_KC_rates\
    #                     + model.dt / model.tau * cs)

    #         MBON_rates = self.rates[self.indices["MBON"],:]
    #         self.output_valence = model.MBONbiases.mm(MBON_rates)
                        

















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