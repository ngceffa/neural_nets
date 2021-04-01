import os
import numpy as np
import matplotlib.pyplot as plt
import time
from torch import from_numpy
from torch.autograd import Variable


def initrandJ(J, scalefac=0.5):
    Jnew = J.astype(np.float32)
    for i in range(Jnew.shape[0]):
        if np.sum(Jnew[i, :] > 0):
            Jnew[i, :] = Jnew[i, :] / np.sqrt(np.sum(Jnew[i, :]**2))
    return scalefac * Jnew

path =\
'/Users/ngc/Documents/GitHub/atg_code/mb_modeling/eschbach2020/refactored_code/'
file_path = path + 'data'

class Connectivity(object):
    """ Neural network weights are derived from synaptic connnectivity.
    """
    def __init__(self, path, cutoff_hemisphere_connections=2, 
                 cutoff_homologous_connections=9):
        csv_connectivity = np.genfromtxt(f'{path}/connectivityMatrix.csv', 
                                              dtype=np.uint8,
                                              delimiter=',',
                                              skip_header=1)
        self.connectivity_total = np.delete(csv_connectivity, 0, axis=1)
        # change pre->post to post->pre
        self.connectivity_total = self.connectivity_total.T 

        self.neurons_names = np.genfromtxt(f'{path}/connectivityMatrix.csv',
                                      dtype=np.unicode_,
                                      delimiter=',',
                                      skip_header=1,
                                      usecols=(0)).tolist()
        self.neurons_names = [name.strip("\"") for name in self.neurons_names]
        #Get indices of left vs. right neurons + single neurons
        left_neurons_indices = \
                               [("LEFT" in name) for name in self.neurons_names]
        right_neurons_indices = \
                              [("RIGHT" in name) for name in self.neurons_names]
        left_neurons_names = \
                                [name.replace(" LEFT", "")
                                for name in self.neurons_names 
                                if ("LEFT" in name)]

        #Get connectivity values (between and within hemispheres)
        connectivity_left_left = self.connectivity_total[
                                                left_neurons_indices, :] \
                                                [:, left_neurons_indices]
        connectivity_right_right = self.connectivity_total\
                                                [right_neurons_indices, :] \
                                                [:, right_neurons_indices]
        connectivity_left_right = self.connectivity_total\
                                                [left_neurons_indices, :] \
                                                [:, right_neurons_indices]
        connectivity_right_left = self.connectivity_total\
                                                [right_neurons_indices, :] \
                                                [:,  left_neurons_indices]
        connectivity_left = connectivity_left_left + connectivity_left_right
        connectivity_right = connectivity_right_left + connectivity_right_right
        # Reduce the connectivity matrix:
        #   - hemi-hemi or hemi-other_hemisphere connections must be 
        #        more than 2 for each neuron
        #   - homologous sum is more than 9 connections
        mask_same_hemisphere = \
                   (connectivity_left_left > cutoff_hemisphere_connections) \
                * (connectivity_right_right > cutoff_hemisphere_connections) 
        mask_opposite_hemispheres = \
                   (connectivity_left_right > cutoff_hemisphere_connections) \
                   * (connectivity_right_left > cutoff_hemisphere_connections)
        hemispheres_mask = mask_same_hemisphere | mask_opposite_hemispheres
        homologous_mask = (connectivity_left + connectivity_right) \
                          > cutoff_homologous_connections
        total_mask = hemispheres_mask * homologous_mask
        # here the starting weights are finally appearing
        self.connectivity = (total_mask) \
                               * (connectivity_left + connectivity_right) / 2
        
        mbons_indices = ["MBON" in name for name in left_neurons_names]
        self.mbons = [name for name in left_neurons_names if "MBON" in name]
        dans_indices = [(("DAN" or "OAN" or "MBIN") in name)\
                        for name in left_neurons_names]
        self.dans = [name for name in left_neurons_names if ("DAN" in name \
                                                    or "OAN" in name \
                                                    or "MBIN" in name)]
        feedback = \
                     [name for name in left_neurons_names if ("FBN" in name \
                                                or "FB2IN" in name \
                                                or "FAN" in name)]
        # subset
        feedback_2 = [name for name in left_neurons_names if "FB2IN" in name]
        feedforward = [name for name in left_neurons_names if "FFN" in name]
        feeds_indices = [(("FBN" or "FB2IN" or "FAN" or "FFN") in name) \
                            for name in left_neurons_names]
        self.feeds = [name for name in left_neurons_names \
                                                if ("FBN" in name \
                                                or "FB2IN" in name \
                                                or "FAN" in name \
                                                or "FFN" in name)]
        mbons_number = len(self.mbons)
        dans_number = len(self.dans)
        feeds_number = len(self.feeds)

        # inhibitory vs excitatory neurons
        with open(f'{path}/inh.txt') as f:
            inhibitory_neurons_names = f.read().split('\n')
        with open(f'{path}/exc.txt') as f:
            excitatory_neurons_names = f.read().split('\n')

        self.inhibitory_mbons_indices = [name in inhibitory_neurons_names \
                                        for name in self.mbons]
        self.inhibitory_feeds_indices = [name in inhibitory_neurons_names \
                                        for name in self.feeds]
        self.excitatory_mbons_indices = [name in excitatory_neurons_names \
                                        for name in self.mbons]
        self.excitatory_feeds_indices = [name in excitatory_neurons_names \
                                        for name in self.feeds]
        

        
        # positive vs negative valence neurons
        with open(f'{file_path}/pipos.txt') as f:
            positive_neurons_names = f.read().split('\n')
        with open(f'{file_path}/pineg.txt') as f:
            negative_neurons_names = f.read().split('\n')

        self.positive_mbons = [name in positive_neurons_names \
                                for name in self.mbons]
        self.negative_mbons = [name in negative_neurons_names \
                                for name in self.mbons]
        # Get compartment names 
        compartments = np.unique([name[-2] for name 
                                           in np.append(self.mbons, self.dans)])
        # Initialise random bias values for MBONs (positive and negative)
        mbons_biases = np.random.randn(1, mbons_number) / np.sqrt(mbons_number)
        mbons_biases[:, self.positive_mbons] = \
                                    np.abs(mbons_biases[:, self.positive_mbons])
        mbons_biases[:, self.negative_mbons] = \
            +                      -np.abs(mbons_biases[:, self.negative_mbons])

        #Get initial connectivity values (from EM; i.e. Jmm0, Jmd0, Jmo0, etc)
        self.J = {}
        indices = [mbons_indices, dans_indices, feeds_indices]   
        neuralnames = ['m', 'd', 'o']

        for preinds, prename in zip(indices, neuralnames):
            for postinds, postname in zip(indices, neuralnames):
                self.J[f'{prename}{postname}0'] = \
                                   self.connectivity[preinds, :][:, postinds]

        Jmm = Variable(from_numpy(initrandJ(self.J["mm0"])), requires_grad=True)
        Jmd = Variable(from_numpy(initrandJ(self.J["md0"])), requires_grad=True)
        Jmo = Variable(from_numpy(initrandJ(self.J["mo0"])), requires_grad=True)
        Jdm = Variable(from_numpy(initrandJ(self.J["dm0"])), requires_grad=True)
        Jdd = Variable(from_numpy(initrandJ(self.J["dd0"])), requires_grad=True)
        Jdo = Variable(from_numpy(initrandJ(self.J["do0"])), requires_grad=True)
        Jom = Variable(from_numpy(initrandJ(self.J["om0"])), requires_grad=True)
        Jod = Variable(from_numpy(initrandJ(self.J["od0"])), requires_grad=True)
        Joo = Variable(from_numpy(initrandJ(self.J["oo0"])), requires_grad=True)
        bm  = Variable(from_numpy(np.zeros(([mbons_number, 1]), 
                                                dtype=np.float32)),
                                                requires_grad=True)
        bd  = Variable(from_numpy(np.zeros(([dans_number, 1]),
                                                dtype=np.float32)),
                                                requires_grad=True)
        bo  = Variable(from_numpy(np.zeros(([feeds_number, 1]),
                                                dtype=np.float32)),
                                                requires_grad=True)
        wdu = Variable(from_numpy(np.random.standard_normal(
                                                [dans_number, 2]) \
                                                .astype(np.float32)), 
                                                requires_grad=True)
        wou = Variable(from_numpy(np.random.standard_normal(
                                                [feeds_number, 2]) \
                                                .astype(np.float32)),
                                                requires_grad=True)
        wrm = Variable(from_numpy(mbons_biases.astype(np.float32)), 
                                                requires_grad=True)
        self.original_train_vars = [Jmm, Jmd, Jmo, Jdm, Jdd, Jdo, Jom, Jod, Joo,
                                    bm, bd, bo,
                                    wdu, wou, wrm]