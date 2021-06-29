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
        batches=30,
        input_size=70,  # Number of Kenyon cells
        input_fraction=0.1,  # KC coding level
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
        self.input_size = input_size
        self.input_fraction = input_fraction
        self.batches = batches
        self.output_size = output_size
        self.dt = dt
        self.timesteps = int(total_time / dt)
        self.total_time = total_time
        self.s0 = np.zeros(
            [self.timesteps, input_size, batches], dtype=np.float32
        )  # KC inputs (time, cell_index)
        self.wmax = 1 / (input_size * input_fraction)
        # Maximum KC-to-MBON synaptic weight
        self.w_init = 0.5 * self.wmax

        self.weights_KC = torch.zeros(
            [self.matrix.shape[0], input_size, batches], dtype=torch.float32
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
            [self.mbons_number], dtype=np.float32
        )
        self.mbin_rates_0 = mbin_initial_rates * np.ones(
            [self.mbins_number], dtype=np.float32
        )
        self.fb_ff_rates_0 = fb_ff_initial_rates * np.ones(
            [self.fb_ff_number], dtype=np.float32
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
            [self.mbons_number, self.input_size], dtype=np.float32  # W (KC->MBON)
        ) / (self.input_size * self.input_fraction)
        self.sbar0 = np.zeros([self.input_size], dtype=np.float32)
        self.dabar0 = np.zeros([self.mbons_number], dtype=np.float32)
        self.ma = np.zeros([self.timesteps, self.mbons_number])
        self.da = np.zeros([self.timesteps, self.mbins_number])
        self.oa = np.zeros([self.timesteps, self.fb_ff_number])
        self.wa = np.zeros(
            [self.timesteps, self.mbons_number, self.input_size, self.batches]
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
        self.lowpass_KC_rates = torch.zeros([self.input_size])

        self.weights_KC = torch.zeros(
            self.n_neurons,
            self.input_size,
        )
        self.weights_KC[self.indices["MBON"], :] = self.w_init

    def apply_constraints(self):
        """ Applies constraints and makes fresh copies of the vars,
        so that multiple instancens of autograd (epochs in training)
        may be done.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        """
        return None


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

                mbon_rates = mbon_rates_new # so that in the formulae the old

                result[ti] = torch.matmul(self.wrm, mbon_rates)

            loss = torch.sum(torch.pow(result - target_output, 2))

            loss.backward(retain_graph=False)  # why retain_graph?
            self.opt.step()
            self.opt.zero_grad()


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
    # model.reset_initial_conditions()
    # model.setNetworkParameters(model.train_vars)
    # already def in Connectivity_3 --> __init__

    odors, valences = nn_utils.odor_valence_pairs(num_cells=70, num_odors=2)
    # n.b. assume all-to-all KCs-MBONs connections

    model.train_first_order(epochs=3)

    # add resting time
    # redo first order...
    # learning rat does not actually change
# %%
