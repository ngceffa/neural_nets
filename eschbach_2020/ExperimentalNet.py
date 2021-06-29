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

    def __init__(
        self,
        path,
        valence=[-1, 1],
        batches=10,
        num_KCs=70,  # Number of Kenyon cells
        active_fraction=0.1,  # KC coding level
        r_size=1,  #
        total_time=80,
        dt=0.5,
        output_size=1,
        reset_times=[30, 60],
        cutoff_hemisphere_connections=2,
        cutoff_homologous_connections=9,
        method="full",  # or 'full'
        condition="control",
        epochs=1,
        mbon_initial_rates=0,
        mbin_initial_rates=0.1,
        fb_ff_initial_rates=0.1,
        tau=5,
        tauw=5,
        ):
        super().__init__(path, cutoff_hemisphere_connections, method=method)
        us_size = len(valence)
        self.mbon_initial_rates = mbon_initial_rates
        self.mbin_initial_rates = mbin_initial_rates
        self.fb_ff_initial_rates = fb_ff_initial_rates
        self.reset_times = [int(reset_times[0] / dt), int(reset_times[1] / dt)]
        self.num_KCs = num_KCs
        self.active_fraction = active_fraction
        self.batches = batches
        self.output_size = output_size
        self.dt = dt
        self.timesteps = int(total_time / dt)
        self.total_time = total_time
        self.s0 = np.zeros(
            [self.timesteps, num_KCs, batches], dtype=np.float32
        )  # KC inputs (time, cell_index)
        self.wmax = 1 / (num_KCs * active_fraction)
        # Maximum KC-to-MBON synaptic weight
        self.w_init = 0.5 * self.wmax
        self.weights_KC_to_mbons = torch.zeros(
            [self.matrix.shape[0], num_KCs, batches], dtype=torch.float32
        )
        # self.weights_KC[self.neuron_idxs["MBON"],:,:] = self.w_init
        self.n_neurons = self.matrix.shape[0]
        self.neuron_types = ["MBON", "all_mbins", "fb_ff"]
        self.rates = torch.zeros(size=(self.n_neurons, self.batches))
        self.learning_rate = 2 * np.logspace(-2, -2, epochs)
        self.opt = torch.optim.RMSprop(self.train_vars, lr=self.learning_rate[0])

        self.DAN_MBON_indices = torch.ones(size=self.matrix.shape)
        for pre in self.indices["all_mbins"]:
            for post in self.indices["MBON"]:
                self.DAN_MBON_indices[post, pre] = 0

        # create a bunch of variables for the simulation
        self.mbon_rates_0 = mbon_initial_rates * np.ones(
            [self.mbons_number, self.batches], dtype=np.float32
        )
        self.mbin_rates_0 = mbin_initial_rates * np.ones(
            [self.mbins_number, self.batches], dtype=np.float32
        )
        self.fb_ff_rates_0 = fb_ff_initial_rates * np.ones(
            [self.fb_ff_number, self.batches], dtype=np.float32
        )
        self.bm = torch.tensor(
            (np.zeros(([self.mbons_number]), dtype=np.float32)), requires_grad=True
        )
        self.bd = torch.tensor(
            (np.zeros(([self.mbins_number]), dtype=np.float32)), requires_grad=True
        )
        self.bf = torch.tensor(
            (np.zeros(([self.fb_ff_number]), dtype=np.float32)), requires_grad=True
        )

        self.tau = tau
        self.tauw = tauw

        self.reset_times = [int(reset_times[0]/self.dt), 
                            int(reset_times[1]/self.dt)]
        self.mbon_rates = torch.from_numpy(mbon_initial_rates * np.ones(
            (self.mbons_number, self.batches), dtype=np.float32
        ))
        self.mbin_rates = torch.from_numpy(mbin_initial_rates * np.ones(
            (self.mbins_number, self.batches), dtype=np.float32
        ))
        self.fb_ff_rates = torch.from_numpy(fb_ff_initial_rates * np.ones(
            (self.fb_ff_number, self.batches), dtype=np.float32
        ))
        # self.baselineinds = (np.sum(self.CS,1,keepdims=True) == 0) \
        #         & (np.sum(self.US,1,keepdims=True) == 0)
        # self.baselineinds_d = torch.from_numpy(np.repeat(\
        #             self.baselineinds, 
        #             self.mbins_number,axis=1).astype(np.float32))

    def reset_initial_conditions(
        self,
        mbon_initial_rates=0,
        mbin_initial_rates=0.1,
        fb_ff_initial_rates=0.1,
        tau=5,  # Timing-dependent plasticity window
        tauw=5,  # Timescale of weight modifications
        ):
        # create a bunch of variables for the simulation
        self.mbon_rates_0 = mbon_initial_rates * np.ones(
            [self.mbons_number], dtype=np.float32
        )
        self.mbin_rates_0 = mbin_initial_rates * np.ones(
            [self.mbins_number], dtype=np.float32
        )
        self.fb_ff_rates_0 = fb_ff_initial_rates * np.ones(
            [self.fb_ff_number], dtype=np.float32
        )
        self.mbon_rates = mbon_initial_rates * np.ones(
            [self.mbons_number], dtype=np.float32
        )
        self.mbin_rates = mbin_initial_rates * np.ones(
            [self.mbins_number], dtype=np.float32
        )
        self.fb_ff_rates = fb_ff_initial_rates * np.ones(
            [self.fb_ff_number], dtype=np.float32
        )

        self.w0 = np.ones(
            [self.mbons_number, self.num_KCs], dtype=np.float32  # W (KC->MBON)
        ) / (self.num_KCs * self.active_fraction)
        self.sbar0 = np.zeros([self.num_KCs], dtype=np.float32)
        self.dabar0 = np.zeros([self.mbons_number], dtype=np.float32)
        self.ma = np.zeros([self.timesteps, self.mbons_number])
        self.da = np.zeros([self.timesteps, self.mbins_number])
        self.oa = np.zeros([self.timesteps, self.fb_ff_number])
        self.wa = np.zeros(
            [self.timesteps, self.mbons_number, self.num_KCs, self.batches]
        )
        self.result = torch.zeros(self.timesteps, self.output_size)
        # output_size is always 1...
        self.dacost = torch.zeros(1)
        for ntype, init_val in zip(
            self.neuron_types,
            [mbon_initial_rates, mbin_initial_rates, fb_ff_initial_rates],
        ):
            self.rates[self.indices[ntype], :] = init_val
        self.lowpass_MBONdopamine = torch.zeros([self.mbons_number])
        self.lowpass_KC_rates = torch.zeros([self.num_KCs])

        self.weights_KC = torch.zeros(
            self.n_neurons,
            self.num_KCs,
        )
        self.weights_KC[self.indices["MBON"], :] = self.w_init

    def reset_initial_conditions_for_batch_training(
        self,
        mbon_initial_rates=0,
        mbin_initial_rates=0.1,
        fb_ff_initial_rates=0.1,
        tau=5,  # Timing-dependent plasticity window
        tauw=5,  # Timescale of weight modifications
        ):
        self.mbon_rates = torch.from_numpy(mbon_initial_rates * np.ones(
            [self.mbons_number, self.batches], dtype=np.float32)
        )
        self.mbin_rates = torch.from_numpy(mbin_initial_rates * np.ones(
            [self.mbins_number, self.batches], dtype=np.float32)
        )
        self.fb_ff_rates = torch.from_numpy(fb_ff_initial_rates * np.ones(
            [self.fb_ff_number, self.batches], dtype=np.float32)
        )
        self.w0 = np.ones(
            [self.mbons_number, self.num_KCs, self.batches], dtype=np.float32  # W (KC->MBON)
        ) / (self.num_KCs * self.active_fraction)
        self.wfast = torch.from_numpy(np.ones(
            [self.mbons_number, self.num_KCs, self.batches], dtype=np.float32
        )) / (self.num_KCs * self.active_fraction)
        self.low_pass_mabons_rate = torch.from_numpy(np.zeros([self.num_KCs, 
            self.batches],  
            dtype=np.float32))
        self.low_pass_mbins_rate = torch.from_numpy(np.zeros([self.mbons_number, 
            self.batches],  
            dtype=np.float32))

    def first_order_trial(self, 
                        valence_values=1, # or -1
                        doUS=True,
                        doC=False,
                        num_KCs=70,
                        active_fraction=0.1, # -> % active KCs
                        total_time=80,
                        dt=0.5,  # timestep
                        cs_duraton=3,
                        us_delay=2,
                        us_duration=3,
                        cs_start_window=[5, 15],
                        cs_test_start_window=[65, 75],
                        returntimes=False
                        ):
        " Random odor is created."
        # CS is a batch of CSs
        self.CS = np.zeros((total_time, num_KCs)) # {0, 1} = {active, inactive}
        self.US = np.zeros((total_time, 2)) # why  2? can we have just {-1, +1, 0} ?
        self.target = np.zeros((total_time))

        active_population = num_KCs * active_fraction
        active_indices = np.random.choice(num_KCs, active_population, replace=False)
        control_indices = np.random.choice(num_KCs, active_population, replace=False)

        stimulus = np.zeros(num_KCs)
        stimulus[active_indices] = 1
        control_stimulus = np.zeros(num_KCs)
        control_stimulus[control_indices] = 1

        cs_start = int(np.random.randint(cs_start_window[0], 
                                        cs_start_window[1]) / self.dt)
        self.CS[start:(cs_start + cs_duraton), :] = stimulus

        us_start = cs_start + us_delay
        test_start = int(np.random.randint(cs_test_start_window[0],
        cs_test_start_window[1]) / self.dt)
        if doC:
            self.CS[test_start:(test_start + cs_duraton), :] = control_stimulus
        else:
            self.CS[test_start:(test_start + cs_duraton),:] = stimulus

        # these 2 entries insted of {-1, 1} I don't like
        # the mbins are hiding one dimension... why?
        if doUS:
            if valence > 0:
                self.US[us_start:(us_start + us_duration), 0] = 1.
            else:
                self.US[us_start:(us_start + us_duration), 1] = 1.

        if doUS and not doC:
            self.target[test_start:(test_start + cs_duraton)] = valence
    
    def first_order_trials(self, 
                        valence_values=[-1, 1],
                        doUS=True,
                        doC=False,
                        batches=10,
                        num_KCs=70,
                        active_fraction=0.1, # -> % active KCs
                        cs_duraton=3,
                        us_delay=2,
                        us_duration=3,
                        cs_start_window=[5, 15],
                        cs_test_start_window=[65, 75],
                        returntimes=False
                        ):
        " Random odor is created."
        # CS is a batch of CSs
        self.CS = np.zeros((self.timesteps, num_KCs, batches))
        self.US = np.zeros((self.timesteps, 2, batches)) 
        self.target = np.zeros((self.timesteps, batches))
        active_population = int(num_KCs * active_fraction)
        
        for b in range(batches):
            
            active_indices = np.random.choice(num_KCs, 
                                            active_population, 
                                            replace=False)
            control_indices = np.random.choice(num_KCs, 
                                            active_population, 
                                            replace=False)
            stimulus = np.zeros(num_KCs)
            stimulus[active_indices] = 1
            control_stimulus = np.zeros(num_KCs)
            control_stimulus[control_indices] = 1
            valence = np.random.choice(valence_values, 1)
            cs_start = int(np.random.randint(cs_start_window[0], 
                                            cs_start_window[1]) / self.dt)
            self.CS[cs_start:(cs_start + cs_duraton), :, b] = stimulus

            us_start = cs_start + us_delay
            test_start = int(np.random.randint(cs_test_start_window[0],
            cs_test_start_window[1]) / self.dt)
            if doC:
                self.CS[test_start:(test_start + cs_duraton), :, b] = control_stimulus
            else:
                self.CS[test_start:(test_start + cs_duraton), :, b] = stimulus

            # these 2 entries insted of {-1, 1} I don't like
            # the mbins are hiding one dimension... why?
            if doUS:
                if valence > 0:
                    self.US[us_start:(us_start + us_duration), 0, b] = 1.
                else:
                    self.US[us_start:(us_start + us_duration), 1, b] = 1.

            if doUS and not doC:
                self.target[test_start:(test_start + cs_duraton), b] = valence
            
        self.CS = torch.from_numpy(self.CS)
        self.US = torch.from_numpy(self.US)
        self.targeet = torch.from_numpy(self.target)

    def temp_evolution(self):
        for t in range(self.timesteps):
            if t in self.reset_times:
                mnew = torch.from_numpy(self.mbon_rates_0)
                dnew = torch.from_numpy(self.mbin_rates_0)
                onew = torch.from_numpy(self.fb_ff_rates_0)
                self.low_pass_mbons_rate = torch.from_numpy(np.zeros([self.num_KCs,
                self.batches],  
                    dtype=np.float32))
                self.low_pass_mbins_rate = torch.from_numpy(np.zeros([
                    self.mbons_number,
                    self.batches],
                    dtype=np.float32))
                # sbar = torch.from_numpy(sbar0)
                # dastdpbar = torch.from_numpy(dastdpbar0)
            else:
                mnew = (1 - self.dt) * self.mbon_rates \
                        + self.dt * torch.relu(torch.tanh(
                            self.J['mm'].mm(self.mbon_rates) 
                            + self.J['mf'].mm(self.fb_ff_rates) 
                            + torch.einsum('ijb,jb->ib',(
                                self.weights_KC_to_mbons, self.CS[t,:,:])) 
                            + self.bm))
                dnew = (1-self.dt)*self.mbin_rates\
                        + self.dt*torch.relu(torch.tanh(
                            self.J['dm'].mm(self.mbon_rates) 
                            + self.J['dd'].mm(self.mbin_rates) 
                            + self.J['df'].mm(self.fb_ff_rates) 
                            + self.w_mbins.mm(self.US[t,:]) 
                            + self.bd))
                onew = (1-self.dt)*self.fb_ff_rates \
                            + self.dt*torch.relu(torch.tanh(
                                self.J['fm'].mm(self.mbon_rates) 
                                + self.J['fd'].mm(self.mbin_rates) 
                                + self.J['ff'].mm(self.fb_ff_rates) 
                                +  self.w_fb_ff.mm(self.US[t,:]) 
                                + self.bf))
            
            self.mbon_rates = mnew
            self.mbin_rates = dnew
            self.fb_ff_rates = onew

            result[t,:,:] = self.wrm.mm(self.mbon_rates)
            dan_mbon_readout = torch.relu(self.J['md'].mm(self.mbin_rates))

            stdp_update = -torch.einsum('ib,jb->ijb',
            (dan_mbon_readout,self.low_pass_mbons_rate)) 
            + torch.einsum('ib,jb->ijb',
            (self.low_pass_mbins_rate, self.CS[t,:,:])) #anti-Hebbian

            wfast = torch.relu(self.w + self.dt*(
                stdp_update - torch.relu(
                    stdp_update - (self.wmax - selfw))))
                     #update that does not exceed wmax
            self.w = self.w + (self.dt/self.tauw)*(wfast - self.w)

            self.low_pass_mbins_rate = (1. - self.dt/self.tau)\
                *self.low_pass_mbins_rate + (self.dt/self.tau)\
                    *dan_mbon_readout
            self.low_pass_mbons_rate = (1. - self.dt/self.tau)*\
                self.low_pass_mbons_ratebar + (self.dt/self.tau)*self.CS[t,:,:]
            
            dacost += torch.sum(torch.pow(torch.relu(
                self.mbin_rates*self.baselineinds_d[t,:,:] - \
                                    self.mbin_initial_rates),2))

    def evolutional_training(self, task='first_order', epochs=1, repetitions=1, 
    savedir='\saved', doplot = False):

        Ntrain = 1
        lossa = np.zeros([Ntrain, epochs])
        lr = 2 * np.logspace(3, -3, Nepochs)
        alphada = .1

        for i in range(Ntrain):

            opt = torch.optim.RMSprop(self.train_vars, lr=lr[0])
            track_loss = np.zeros(epochs)
            track_loss_da = np.zeros(epochs)
            self.w = torch.from_numpy(self.w0)
            lastt = time.time()
            
            for epoch in range(epochs):
                for group in opt.param_groups:
                    group['lr'] = lr[epoch] # change gradient descent step
                self.reset_initial_conditions_for_batch_training()
                self.first_order_trials() # s, u, rtarg
                self.baselineinds = (np.sum(self.CS,1,keepdims=True) == 0) \
                & (np.sum(self.US,1,keepdims=True) == 0)
                self.baselineinds_d = torch.from_numpy(np.repeat(\
                    self.baselineinds, 
                    self.mbins_number,axis=1).astype(np.float32))
                
                result = torch.zeros(self.timesteps, 1, self.batches)
                dacost = torch.zeros(1) # ?

    def train_first_order(self, task="first_order", epochs=1, show=False):

        self.reset_initial_conditions()

        # Variables common for every epoch
        self.loss = np.zeros((epochs))
        self.learning_rate = 2 * np.logspace(-3, -3, epochs)
        alphada = 0.1  # ?
        modelfname = os.getcwd() + "first_order.pt"

        # opt = torch.optim.RMSprop(self.train_vars, lr=self.learning_rate[0])
        self.track_loss = np.zeros(epochs)
        self.track_loss_da = np.zeros(epochs)
        odors, valences = nn_utils.odor_valence_pairs(num_cells=70, num_odors=1)

        mbon_rates = torch.from_numpy(self.mbon_rates_0)
        mbon_rates.requires_grad = False
        mbin_rates = torch.from_numpy(self.mbin_rates_0)
        mbin_rates.requires_grad = False
        fb_ff_rates = torch.from_numpy(self.fb_ff_rates_0)
        fb_ff_rates.requires_grad = False

        w = torch.from_numpy(self.w0)  # W (KC->MBON), mbons post, KCs pre
        wfast = torch.from_numpy(self.w0)
        low_pass_mbon_rates = torch.from_numpy(self.sbar0)
        low_pass_mbin_rates = torch.from_numpy(self.dabar0)

        track_loss = []
        track_loss_da = []

        start = time.time()

        ciao = self.J['mm'].data.numpy()
        ciao = torch.from_numpy(ciao)
        ciao_new = self.J['mm'].data.numpy()

        dacost = torch.zeros(1)  # ???
        for epoch in range(epochs):

            self.opt = torch.optim.RMSprop(self.train_vars, lr=self.learning_rate[0])
            mbon_rates = torch.from_numpy(self.mbon_rates_0)
            mbon_rates.requires_grad = False
            mbin_rates = torch.from_numpy(self.mbin_rates_0)
            mbin_rates.requires_grad = False
            fb_ff_rates = torch.from_numpy(self.fb_ff_rates_0)
            fb_ff_rates.requires_grad = False

            ciao = torch.from_numpy(ciao_new)
            ciao.requires_grad_()
    
            for g in self.opt.param_groups:
                g["lr"] = self.learning_rate[epoch]

            KCs, us, target_output = nn_utils.first_order_single_xp(
                odors[:, 0], 1, total_time=self.total_time, dt=self.dt
            )

            baselineinds = (np.sum(nn_utils.tens2np(KCs), 1, keepdims=True) == 0) \
                        & (np.sum(nn_utils.tens2np(us), keepdims=True) == 0
            )
            baselineinds_d = torch.from_numpy(
                np.repeat(baselineinds, self.mbins_number, axis=1).astype(np.float32)
            )
            result = torch.zeros(self.timesteps)

            self.train_vars[0] = ciao

            for ti in range(self.timesteps):

                mbon_rates_new = (1 - self.dt) * mbon_rates \
                    + self.dt * torch.relu(
                    torch.tanh(
                        torch.matmul(ciao, mbon_rates)
                        + torch.matmul(self.J["mf"].data, fb_ff_rates)
                        + torch.matmul(self.J["md"].data, mbin_rates)
                        + torch.einsum("ij,j->i", (w.float(), KCs[ti, :].float()))
                        + self.bm
                    )
                )
                mbin_rates_new = (1 - self.dt) * mbin_rates \
                    + self.dt * torch.relu(
                    torch.tanh(
                        torch.matmul(self.J["dm"].data, mbon_rates)
                        + torch.matmul(self.J["dd"].data, mbin_rates)
                        + torch.matmul(self.J["df"].data, fb_ff_rates)
                        + torch.matmul(self.w_mbins.float(), us[ti, :].float())
                        # ext input from US
                        + self.bd
                    )
                )
                fb_ff_rates_new = (1 - self.dt) * fb_ff_rates  \
                    + self.dt * torch.relu(
                    torch.tanh(
                        torch.matmul(self.J["fm"].data, mbon_rates)
                        + torch.matmul(self.J["fd"].data, mbin_rates)
                        + torch.matmul(self.J["ff"].data, fb_ff_rates)
                        + torch.matmul(self.w_fb_ff.float(), us[ti, :].float())
                        # ext input from US
                        + self.bf
                    )
                )

                mbon_rates = mbon_rates_new # so that in the formulae the old
                # values are used before updating
                mbin_rates = mbin_rates_new
                fb_ff_rates = fb_ff_rates_new

                result[ti] = torch.matmul(self.wrm, mbon_rates)

                # "dopamine" level in MBONS
                mbin_to_mbon = torch.relu(
                    torch.matmul(self.J["md"].data, mbin_rates)
                )

                stdp_update = -torch.einsum(
                    "i,j->ij", (mbin_to_mbon, low_pass_mbon_rates)
                ) + torch.einsum("i,j->ij", (low_pass_mbin_rates, KCs[ti, :]))
                # anti-Hebbian

                wfast = torch.relu(
                    w
                    + self.dt
                    * (stdp_update - torch.relu(stdp_update - (self.wmax - w)))
                )
                # update that does not exceed wmax
                w = w + (self.dt / self.tauw) * (wfast - w)  # eq 3

                low_pass_mbin_rates = (
                    1.0 - self.dt / self.tau
                ) * low_pass_mbin_rates + (self.dt / self.tau) * torch.relu(
                    torch.matmul(self.J["md"].detach(), mbin_rates)
                )
                low_pass_mbon_rates = (
                    1.0 - self.dt / self.tau
                ) * low_pass_mbon_rates + (self.dt / self.tau) * KCs[ti, :]

                dacost += torch.sum(
                    torch.pow(
                        torch.relu(
                            mbin_rates * baselineinds_d[ti, :] - self.mbon_initial_rates
                        ),
                        2,
                    )
                )
            loss_err = torch.sum(torch.pow(result - target_output, 2))
            loss_da = alphada * dacost
            loss = loss_err + loss_da
            track_loss.append(loss_err.detach().numpy())
            track_loss_da.append(loss_da)
            loss.backward(retain_graph=False)  # why retain_graph?
            self.opt.step()
            self.opt.zero_grad()

            # #sparse connectivity -> keep zeros at zero
            # THESE MODIFICATIONS ARE NOT "IN PLACE

            self.J["mm"].data *= (self.W["mm0"] != 0)
            self.J["md"].data *= self.W["md0"] != 0
            self.J["mf"].data *= self.W["mf0"] != 0
            self.J["dm"].data *= self.W["dm0"] != 0
            self.J["dd"].data *= self.W["dd0"] != 0
            self.J["df"].data *= self.W["df0"] != 0
            self.J["fm"].data *= self.W["fm0"] != 0
            self.J["fd"].data *= self.W["fd0"] != 0
            self.J["ff"].data *= self.W["ff0"] != 0

            # Remember "post<-pre" = "row-col"
            # check again these constraints...
            # nbb post-pre is an important distinction

            self.J["mm"].data[:, self.indices["excitatory_mbon"]] = torch.relu(
                self.J["mm"].data[:, self.indices["excitatory_mbon"]]
            )
            self.J["dm"].data[:, self.indices["excitatory_mbon"]] = torch.relu(
                self.J["dm"].data[:, self.indices["excitatory_mbon"]]
            )
            self.J["fm"].data[:, self.indices["excitatory_mbon"]] = torch.relu(
                self.J["fm"].data[:, self.indices["excitatory_mbon"]]
            )
            self.J["mf"].data[:, self.indices["excitatory_fb_ff"]] = torch.relu(
                self.J["mf"].data[:, self.indices["excitatory_fb_ff"]]
            )
            self.J["df"].data[:, self.indices["excitatory_fb_ff"]] = torch.relu(
                self.J["df"].data[:, self.indices["excitatory_fb_ff"]]
            )
            self.J["ff"].data[:, self.indices["excitatory_fb_ff"]] = torch.relu(
                self.J["ff"].data[:, self.indices["excitatory_fb_ff"]]
            )

            self.J["mm"].data[:, self.indices["inhibitory_mbon"]] = -torch.relu(
                (-self.J["mm"].data[:, self.indices["inhibitory_mbon"]])
            )
            self.J["dm"].data[:, self.indices["inhibitory_mbon"]] = -torch.relu(
                (-self.J["dm"].data[:, self.indices["inhibitory_mbon"]])
            )
            self.J["fm"].data[:, self.indices["inhibitory_mbon"]] = -torch.relu(
                (-self.J["fm"].data[:, self.indices["inhibitory_mbon"]])
            )
            self.J["mf"].data[:, self.indices["inhibitory_fb_ff"]] = -torch.relu(
                (-self.J["mf"].data[:, self.indices["inhibitory_fb_ff"]])
            )
            self.J["df"].data[:, self.indices["inhibitory_fb_ff"]] = -torch.relu(
                (-self.J["df"].data[:, self.indices["inhibitory_fb_ff"]])
            )
            self.J["ff"].data[:, self.indices["inhibitory_fb_ff"]] = -torch.relu(
                (-self.J["ff"].data[:, self.indices["inhibitory_fb_ff"]])
            )

            # plt.plot((result.data.numpy()/np.mean(result.data.numpy())))
            # plt.plot(target_output, lw=.5, color='r')
            # plt.show()

            self.wrm.data[self.indices["positive_mbon"]] = torch.relu(
                self.wrm.data[self.indices["positive_mbon"]]
            )
            self.wrm.data[self.indices["negative_mbon"]] = -torch.relu(
                -self.wrm.data[self.indices["negative_mbon"]]
            )
            data = self.J['mm'].data.numpy()
            self.J['mm'] = nn_utils.from_np(data)
            self.train_vars = [ciao, self.J['md'], self.J['mf'],
                                    self.J['dm'], self.J['dd'], self.J['df'],
                                    self.J['fm'], self.J['fd'],  self.J['ff'],
                                    self.bm, self.bd, self.bf,
                                    self.w_mbins, self.w_fb_ff, self.wrm]

            # come se non esistessero negative mbons
            # acquisto una baseline e me la tengo per sempre...
            print(loss)
            #devi togliere i "data" e ricreare loss/optper ogni epoch
            # risalvando correttamente i valori nelle variabili...

            

            plt.imshow(ciao.data.numpy())
            plt.show()
            ciao_new = ciao.data.numpy()

        print(time.time() - start)


if __name__ == "__main__":
    pass

    # plt.style.use('grayscale')

    file_path = os.getcwd() + "/eschbach_2020/data"

    model = MBmodel(file_path)
    model.first_order_trials()
    print(model.CS.shape, model.US.shape, model.target.shape)
    print(type(model.CS))
    #model.temp_evolution()
    a = torch.einsum('ijb,jb->ib',(
                    model.weights_KC_to_mbons.float(), model.CS[0,:,:].float()))