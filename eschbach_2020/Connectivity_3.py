import os
import numpy as np
import torch
from torch import from_numpy
from torch.autograd import Variable
import nn_utility_functions as nn_utils
import matplotlib.pyplot as plt


class Connectivity(object):
    """ It acts as organized container for data from connectivityMatrix.csv. 
    Neural network weights are derived from synaptic connnectivity.
    Depenng on the condition, more modifications are made to the weights.
    Dpendind on the method:
        - ashock_2020 -> cut in same emisphere, and then in cross-emisphere;
            use the average of the connection (size = 1/2 of start)
        - full -> jsut cut out Autapses and use all the connections;
            n.b. the two single neurons are still used in this case.
    """
    def __init__(self,
                 path,
                 cutoff_hemisphere_connections=2,
                 cutoff_homologous_connections=9,
                 cutoff_autapses=3,
                 method='full',
                 condition='control'):

        # inhibitory vs excitatory neurons
        with open(f'{path}/inh.txt') as f:
            inhibitory_neurons_names = f.read().split('\n')
            assert inhibitory_neurons_names
        with open(f'{path}/exc.txt') as f:
            excitatory_neurons_names = f.read().split('\n')
            assert excitatory_neurons_names
        # positive vs negative valence neurons
        with open(f'{path}/pipos.txt') as f:
            positive_neurons_names = f.read().split('\n')
            assert positive_neurons_names
        with open(f'{path}/pineg.txt') as f:
            negative_neurons_names = f.read().split('\n')
        assert negative_neurons_names

        # here is the choice of method:
        if method == 'ashock_2020':
            self.neurons_names, self.matrix = \
                                        nn_utils.filtered_average_connectivity(
                                        f'{path}',
                                        cutoff_hemisphere_connections,
                                        cutoff_homologous_connections)
        elif method == 'full':
             self.neurons_names, self.matrix = nn_utils.\
                                        both_hemispheres_connectivity(f'{path}',
                                                cutoff_hemisphere_connections)
        else:
            print('method parameter is wrong.\n')
            return None

        self.neurons_types = np.unique([name.split('-')[0]
                                                for name in self.neurons_names])

        # ----------------------------------------------------------------------
        # Define a dictionary for the names
        self.names = {}
        self.names['mbon'] = \
                         [name for name in self.neurons_names if "MBON" in name]
        self.names['mbin'] = [name for name in self.neurons_names
                          if ("DAN" in name or "OAN" in name or "MBIN" in name)]
        self.names['fb_ff'] = [name for name in self.neurons_names
                                                if ("FBN" in name
                                                or "FB2IN" in name
                                                or "FAN" in name
                                                or "FFN" in name)]
        self.mbons_number = len(self.names['mbon'])
        self.mbins_number = len(self.names['mbin'])
        self.fb_ff_number = len(self.names['fb_ff'])
        # redefined so that they are sorted
        self.sorted_neurons_names = np.r_[self.names['mbon'],
                                  self.names['mbin'],
                                  self.names['fb_ff']]

        # ----------------------------------------------------------------------
        # Define a dictionary with indices: these can retrieve the subset of 
        # connections in the original self.matrix 
        self.indices = {}
        # all kinds of neurons:{'DAN', 'FAN', 'FB2IN', 'FBN', 'FFN', 'MBIN',
        #                       'MBON', 'OAN'}
        # n.b. case sensitive
        for kind in self.neurons_types:
            self.indices[kind] = [kind in name for name in self.neurons_names]
        # Add subsets:
        self.indices['fb'] = ["FBN" in name
                             or "FB2IN" in name
                             or "FAN" in name
                             for name in self.neurons_names]
        self.indices['fb_ff'] = ["FBN" in name
                                or "FB2IN" in name
                                or "FAN" in name
                                or "FFN" in name
                                for name in self.neurons_names]
        self.indices['all_mbins'] = ["DAN" in name
                                    or "OAN" in name
                                    or "MBIN" in name
                                    for name in self.neurons_names]
        self.indices['inhibitory_mbon'] = [name in inhibitory_neurons_names \
                                        for name in self.names['mbon']]
        self.indices['inhibitory_fb_ff']= [name in inhibitory_neurons_names \
                                        for name in self.names['fb_ff']]
        self.indices['excitatory_mbon'] = [name in excitatory_neurons_names \
                                        for name in self.names['mbon']]
        self.indices['excitatory_fb_ff'] = [name in excitatory_neurons_names \
                                        for name in self.names['fb_ff']]
        self.indices['positive_mbon'] = [name in positive_neurons_names \
                                        for name in self.names['mbon']]
        self.indices['negative_mbon'] = [name in negative_neurons_names \
                                        for name in self.names['mbon']]
        # ----------------------------------------------------------------------

        # Get compartment names 
        self.compartments = np.unique([name[-2] for name
                    in np.append(self.names['mbon'], self.names['mbin'])])
        for comp_name in self.compartments:
            self.indices[comp_name] = [comp_name in name[-2] \
                                        for name in self.sorted_neurons_names]

        # Get initial connectivity values (from EM; i.e. Wmm0, Wmd0, Wmo0, etc)
        self.W = {}
        self.J = {}
        indices = [self.indices['MBON'], 
                   self.indices['all_mbins'],
                   self.indices['fb_ff']]
        neurons_categories = ['m', 'd', 'f'] # grouped by class:
                                             # m = mbons; 
                                             # d = dans, oans, mbins;
                                             # f = ffn, fbn, fb2n, fan
        for preinds, prename in zip(indices, neurons_categories):
            for postinds, postname in zip(indices, neurons_categories):
                # original
                self.W[f'{prename}{postname}0'] = \
                    torch.tensor(nn_utils.initrandW(
                     self.matrix[preinds, :][:, postinds]), requires_grad=True)
                # to be trained
                self.J[f'{prename}{postname}'] = \
                    torch.tensor(nn_utils.initrandW(
                     self.matrix[preinds, :][:, postinds]), requires_grad=True)
                

        # Apply the condition --> it usually excludes some connections
        if condition == "control":
            pass
        elif condition == "control2":
            self.W['mm0'][:, :] = 0
            self.W['dm0'][:, :] = 0
        elif condition == "nofb": # this also cuts the feedforward...
            self.W['ff0'][:, :] = 0
            self.W['mf0'][:, ] = 0
            self.W['df0'][:, :] = 0
            self.W['dm0'][:, :] = 0
            self.W['dd0'][:, :] = 0
        elif condition == "nofbns":
            self.W['ff0'][:, :] = 0
            self.W['mf0'][:, :] = 0
            self.W['df0'][:, :] = 0
        elif condition == "nofb2ins":
            self.W['ff0'][:, self.indices['fb2inn']] = 0
            self.W['mf0'][:, self.indices['fb2inn']] = 0
            self.W['df0'][:, self.indices['fb2inn']] = 0
        elif condition == "no2step":
            self.W['ff0'][:, :] = 0
            self.W['mo0'][:, :] = 0
            self.W['df0'][:, self.indices['fb2inn']] = 0
        elif condition == "no2step_dirind":
            self.W['ff0'][:, :] = 0
            self.W['mf0'][:, :] = 0
            self.W['df0'][:, self.indices['fb2inn']] = 0
            self.W['mm0'][:, :] = 0
            self.W['dm0'][:, :] = 0
            # there are more in the original code
        else:
            print("WARNING: INVALID CONDITION --> applying 'control' condition")
        self.condition = condition

        mbons_biases = np.random.randn(1, self.mbons_number) / \
                                        np.sqrt(self.mbons_number)
        mbons_biases[:, self.indices['positive_mbon']] = \
                          np.abs(mbons_biases[:, self.indices['positive_mbon']])
        mbons_biases[:, self.indices['negative_mbon']] = \
                        -np.abs(mbons_biases[:, self.indices['negative_mbon']])
        # Biases
        self.bm  = torch.tensor((np.zeros(([self.mbons_number, 1]),
                                                dtype=np.float32)),
                                                requires_grad=True)
        self.bd  = torch.tensor((np.zeros(([self.mbins_number, 1]),
                                                dtype=np.float32)),
                                                requires_grad=True)
        self.bf  = torch.tensor((np.zeros(([self.fb_ff_number, 1]),
                                                dtype=np.float32)),
                                                requires_grad=True)
        # model variables (?) <-- come back
        self.w_mbins = torch.tensor((np.random.standard_normal(
                                                [self.mbins_number, 2]) \
                                                .astype(np.float32)),
                                                requires_grad=True)
        self.w_fb_ff = torch.tensor((np.random.standard_normal(
                                                [self.fb_ff_number, 2]) \
                                                .astype(np.float32)),
                                                requires_grad=True)
        # This is used for MBONS readout --> valence thy compute
        self.wrm = torch.tensor((mbons_biases.astype(np.float32)),
                                                requires_grad=True)
        
        self.train_vars = [self.W['mm0'], self.W['md0'], self.W['mf0'],
                            self.W['dm0'], self.W['dd0'], self.W['df0'],
                            self.W['fm0'], self.W['fd0'],  self.W['ff0'],
                            self.bm, self.bd, self.bf,
                            self.w_mbins, self.w_fb_ff, self.wrm]
        
        self.weights = torch.tensor(
                        np.vstack((
                        (np.hstack((
                        nn_utils.tens2np(self.W['mm0']), 
                        nn_utils.tens2np(self.W['md0']),
                        nn_utils.tens2np(self.W['mf0'])
                        )),\
                        np.hstack((
                        nn_utils.tens2np(self.W['dm0']), 
                        nn_utils.tens2np(self.W['dd0']),
                        nn_utils.tens2np(self.W['df0'])
                        )),\
                        np.hstack((
                        nn_utils.tens2np(self.W['fm0']), 
                        nn_utils.tens2np(self.W['fd0']),
                        nn_utils.tens2np(self.W['ff0'])
                        ))))), requires_grad=True)
        self.biases = torch.tensor(
                        np.vstack((
                        nn_utils.tens2np(self.bm),
                        nn_utils.tens2np(self.bd),
                        nn_utils.tens2np(self.bf)
                        )), requires_grad=True)
        self.us_weights = torch.tensor(
                            np.vstack((
                            nn_utils.tens2np((torch.zeros((
                            self.mbons_number + self.mbins_number, 2)))),
                            nn_utils.tens2np(self.w_fb_ff)
                            )), requires_grad=True)
        self.MBONbiases = self.wrm.detach().requires_grad=True # meaning?

        self.train_vars_grouped = [self.weights,
                                   self.biases,
                                   self.us_weights,
                                   self.MBONbiases]

        # THese are already inside somewhere:
        # self.allowedConnections = np.array(weights!=0, dtype=int)
        # self.allowedUSWeights = np.array(us_weights!=0, dtype=int)
    

    # def setNetworkParametersFromOriginal(self,train_vars):
    #     Jmm, Jmd, Jmo, Jdm, Jdd, Jdo, Jom, Jod, Joo, bm, bd, bo, wdu, wou, wrm = train_vars
    #     self.weights = from_np(np.vstack([ # post -> pre
    #         np.hstack(tens2np([Jmm, Jmd, Jmo])),
    #         np.hstack(tens2np([Jdm, Jdd, Jdo])),
    #         np.hstack(tens2np([Jom, Jod, Joo]))]))
    #     self.biases = from_np(np.vstack(tens2np([bm,bd,bo])))
    #     self.us_weights = fill_us_weights(wou)
    #     self.MBONbiases = wrm.detach()

if __name__=="__main__":
    #plt.style.use('classic')
    print('CWD: ', os.getcwd())
    file_path = os.getcwd() + '/eschbach_2020/data'
    conn = Connectivity(file_path, method='full')
    fig = plt.figure(2)
    showme = conn.matrix[conn.indices['MBON'], :][:, conn.indices['all_mbins']]
    fig.add_subplot(131)
    plt.imshow(showme, interpolation='none', aspect = 'auto')
    fig.add_subplot(132)
    plt.plot(np.sum(showme, axis=0))
    fig.add_subplot(133)
    plt.plot(np.sum(showme, axis=1))
    plt.show()