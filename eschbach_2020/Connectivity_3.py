import os
import numpy as np
import torch
from torch import from_numpy
from torch.autograd import Variable
import nn_utility_functions as nn_utils
import matplotlib.pyplot as plt
import pprint


class Connectivity(object):
    """It acts as organized container for data from connectivityMatrix.csv.
    Neural network weights are derived from synaptic connnectivity.
    Depenng on the condition, more modifications are made to the weights.
    Dependind on the method:
        - ashock_2020 -> cut in same emisphere, and then in cross-emisphere;
            use the average of the connection (size = 1/2 of start)
        - full -> just cut out Autapses and use all the connections;
            n.b. there are two uncoupled neurons still used in this case.
    """

    def __init__(
        self,
        path,
        cutoff_hemisphere_connections=2,
        cutoff_homologous_connections=9,
        cutoff_autapses=3,
        method="full",
        condition="control",
        print_dict_keys=False,
    ):

        # open files with known info:{excitator/inhibitory, positive/negative}
        with open(f"{path}/inh.txt") as f:
            inhibitory_neurons_names = f.read().split("\n")
            assert inhibitory_neurons_names
        with open(f"{path}/exc.txt") as f:
            excitatory_neurons_names = f.read().split("\n")
            assert excitatory_neurons_names
        with open(f"{path}/pipos.txt") as f:
            positive_neurons_names = f.read().split("\n")
            assert positive_neurons_names
        with open(f"{path}/pineg.txt") as f:
            negative_neurons_names = f.read().split("\n")
            assert negative_neurons_names

        if method == "ashock_2020":
            self.neurons_names, self.matrix = nn_utils.filtered_average_connectivity(
                f"{path}", cutoff_hemisphere_connections, cutoff_homologous_connections
            )
        elif method == "full":
            self.neurons_names, self.matrix = nn_utils.both_hemispheres_connectivity(
                f"{path}", cutoff_hemisphere_connections
            )
        elif method == "full_with_autapses":
            self.neurons_names, self.matrix = nn_utils.both_hemispheres_connectivity_allow_autapses(
                f"{path}", cutoff_hemisphere_connections
            )
        else:
            print("Method parameter is wrong.\n")
            return None
        self.neurons_types = np.unique(
            [name.split("-")[0] for name in self.neurons_names]
        )
        self.neurons_names_no_left_right = [
            name.split(" ")[0] for name in self.neurons_names
        ]

        self.names = {}
        self.names["mbon"] = [name for name in self.neurons_names if "MBON" in name]
        self.names["mbin"] = [
            name
            for name in self.neurons_names
            if ("DAN" in name or "OAN" in name or "MBIN" in name)
        ]
        self.names["fb_ff"] = [
            name
            for name in self.neurons_names
            if ("FBN" in name or "FB2IN" in name or "FAN" in name or "FFN" in name)
        ]
        self.mbons_number = len(self.names["mbon"])
        self.mbins_number = len(self.names["mbin"])
        self.fb_ff_number = len(self.names["fb_ff"])
        self.sorted_neurons_names = np.r_[
            self.names["mbon"], self.names["mbin"], self.names["fb_ff"]
        ]
        # cancel left/right in the names
        self.names["mbon_no_lr"] = [name.split(" ")[0] for name in self.names["mbon"]]
        self.names["mbin_no_lr"] = [name.split(" ")[0] for name in self.names["mbin"]]
        self.names["fb_ff_no_lr"] = [name.split(" ")[0] for name in self.names["fb_ff"]]

        # Indices to retrieve the subset of connections from self.matrix.
        self.indices = {}
        # all kinds of neurons
        # {'DAN', 'FAN', 'FB2IN', 'FBN', 'FFN', 'MBIN','MBON', 'OAN'}
        # n.b. case sensitive
        for kind in self.neurons_types:
            self.indices[kind] = [kind in name for name in self.neurons_names]
        # Add subsets:
        self.indices["fb"] = [
            "FBN" in name or "FB2IN" in name or "FAN" in name
            for name in self.neurons_names
        ]
        self.indices["fb_ff"] = [
            "FBN" in name or "FB2IN" in name or "FAN" in name or "FFN" in name
            for name in self.neurons_names
        ]
        self.indices["all_mbins"] = [
            "DAN" in name or "OAN" in name or "MBIN" in name
            for name in self.neurons_names
        ]
        self.indices["inhibitory_mbon"] = [
            name in inhibitory_neurons_names for name in self.names["mbon_no_lr"]
        ]
        self.indices["inhibitory_fb_ff"] = [
            name in inhibitory_neurons_names for name in self.names["fb_ff_no_lr"]
        ]
        self.indices["excitatory_mbon"] = [
            name in excitatory_neurons_names for name in self.names["mbon_no_lr"]
        ]
        self.indices["excitatory_fb_ff"] = [
            name in excitatory_neurons_names for name in self.names["fb_ff_no_lr"]
        ]
        self.indices["positive_mbon"] = [
            name in positive_neurons_names for name in self.names["mbon_no_lr"]
        ]
        self.indices["negative_mbon"] = [
            name in negative_neurons_names for name in self.names["mbon_no_lr"]
        ]

        # Get compartment names
        self.compartments = np.unique(
            [
                name[-2:]
                for name in np.append(
                    self.names["mbon_no_lr"], self.names["mbin_no_lr"]
                )
            ]
        )

        for comp_name in self.compartments:
            self.indices[comp_name] = [
                comp_name in name for name in self.neurons_names
            ]

        # Get initial connectivity values (from EM; i.e. Wmm0, Wmd0...),
        # and varibale copies for Autograd (Jmm, Jmd...).
        self.W = {}
        self.J = {}
        indices = [
            self.indices["MBON"],
            self.indices["all_mbins"],
            self.indices["fb_ff"],
        ]
        neurons_categories = ["m", "d", "f"]
        # For concise notation purpose only, grouped by "class":
        # m = mbons;
        # d = dans, oans, mbins; (dopaminergic, octopaminergic, general);
        # f = ffn, fbn, fb2n, fan (feedback, feedforward).
        for preinds, prename in zip(indices, neurons_categories):
            for postinds, postname in zip(indices, neurons_categories):
                # original
                self.W[f"{prename}{postname}0"] = \
                    (self.matrix[preinds, :][:, postinds])
                
                # to be trained
                self.J[f"{prename}{postname}"] = torch.tensor(
                    nn_utils.initrandW(
                        self.matrix[preinds, :][:, postinds], scalefac=0.5
                    ),
                    requires_grad=True,
                )

        self.J["mm"].data[:, self.indices["inhibitory_mbon"]] *= -1
        self.J["dm"].data[:, self.indices["inhibitory_mbon"]] *= -1
        self.J["fm"].data[:, self.indices["inhibitory_mbon"]] *= -1
        self.J["mf"].data[:, self.indices["inhibitory_fb_ff"]] *= -1
        self.J["df"].data[:, self.indices["inhibitory_fb_ff"]] *= -1
        self.J["ff"].data[:, self.indices["inhibitory_fb_ff"]] *= -1

        # Apply the condition --> it usually excludes some connections
        # TODO: propagate condition onto Jmm, Jmd...
        if condition == "control":
            pass
        elif condition == "control2":
            self.W["mm0"][:, :] = 0
            self.W["dm0"][:, :] = 0
        elif condition == "nofb":  # this also cuts the feedforward...
            self.W["ff0"][:, :] = 0
            self.W["mf0"][
                :,
            ] = 0
            self.W["df0"][:, :] = 0
            self.W["dm0"][:, :] = 0
            self.W["dd0"][:, :] = 0
        elif condition == "nofbns":
            self.W["ff0"][:, :] = 0
            self.W["mf0"][:, :] = 0
            self.W["df0"][:, :] = 0
        elif condition == "nofb2ins":
            self.W["ff0"][:, self.indices["fb2inn"]] = 0
            self.W["mf0"][:, self.indices["fb2inn"]] = 0
            self.W["df0"][:, self.indices["fb2inn"]] = 0
        elif condition == "no2step":
            self.W["ff0"][:, :] = 0
            self.W["mo0"][:, :] = 0
            self.W["df0"][:, self.indices["fb2inn"]] = 0
        elif condition == "no2step_dirind":
            self.W["ff0"][:, :] = 0
            self.W["mf0"][:, :] = 0
            self.W["df0"][:, self.indices["fb2inn"]] = 0
            self.W["mm0"][:, :] = 0
            self.W["dm0"][:, :] = 0
            # there are more in the original code
        else:
            print("WARNING: INVALID CONDITION --> applying 'control' condition")
        self.condition = condition

        mbons_readout_weights = np.random.randn(self.mbons_number) / (self.mbons_number)
        mbons_readout_weights[self.indices["positive_mbon"]] = np.abs(
            mbons_readout_weights[self.indices["positive_mbon"]]
        )
        mbons_readout_weights[self.indices["negative_mbon"]] = -np.abs(
            mbons_readout_weights[self.indices["negative_mbon"]]
        )
        # Concise notation for the biases. To be trained.
        self.bm = torch.tensor(
            (np.zeros(([self.mbons_number, 1]), dtype=np.float32)), requires_grad=True
        )
        self.bd = torch.tensor(
            (np.zeros(([self.mbins_number, 1]), dtype=np.float32)), requires_grad=True
        )
        self.bf = torch.tensor(
            (np.zeros(([self.fb_ff_number, 1]), dtype=np.float32)), requires_grad=True
        )
        # model variables (?) <-- come back
        self.w_mbins = torch.tensor(
            (np.random.standard_normal([self.mbins_number, 2]).astype(np.float32)),
            requires_grad=True,
        )
        self.w_fb_ff = torch.tensor(
            (np.random.standard_normal([self.fb_ff_number, 2]).astype(np.float32)),
            requires_grad=True,
        )
        # MBONS readout --> to compute what valence they show
        self.wrm = torch.tensor(
            (mbons_readout_weights.astype(np.float32)), requires_grad=True
        )
        # To easy pass all variables to the torch optimizar
        self.train_vars = [self.J['mm'], self.J['md'], self.J['mf'],
                                    self.J['dm'], self.J['dd'], self.J['df'],
                                    self.J['fm'], self.J['fd'],  self.J['ff'],
                                    self.bm, self.bd, self.bf,
                                    self.w_mbins, self.w_fb_ff, self.wrm]

        if print_dict_keys:
            print("\nTwo dictionaries have been defined.\n")
            print("1) Names containing list of names with keys: \n")
            print(self.names.keys(), sep=" \t")
            print("\n2) Indices (0, 1 list values), with keys:\n")
            print(self.indices.keys(), sep=" \t")
            print("\n")


if __name__ == "__main__":

    print("CWD: ", os.getcwd())
    file_path = os.getcwd() + "/eschbach_2020/data"
    file_path = '/home/ngc/Documents/GitHub/neural_nets/eschbach_2020/data'
    conn = Connectivity(file_path, method="full_with_autapses", print_dict_keys=False)
    conn_avg = Connectivity(file_path, method='ashock_2020')
    # plt.imshow((conn.J['mm'].data.numpy()))
    # plt.show()
    # fig = plt.figure('recap full')
    # fig.add_subplot(221)
    # plt.title('scaled')
    # plt.imshow(conn.J['dd'].data.numpy(), interpolation='none')
    # plt.xlabel('mbons')
    # plt.ylabel('mbons')
    # fig.add_subplot(222)
    # plt.title('original')
    # plt.imshow(conn.W['dd0']**.5, interpolation='none')
    # plt.grid()

    # fig.add_subplot(223)
    # plt.title('avg_scaled')
    # plt.imshow(conn_avg.J['dd'].data.numpy(), interpolation='none')
    # plt.xlabel('mbons')
    # plt.ylabel('mbons')
    # fig.add_subplot(224)
    # plt.title('avg_original')
    # plt.imshow(conn_avg.W['dd0']**.5, interpolation='none')
    # plt.grid()
    # plt.xlabel('mbins')
    # plt.show()

    # check strange symmetry in DANS


    dan_top_full = conn.W['mm0']
    dan_bot_full = conn.W['mm0']

    asymm_index = nn_utils.symmetry_index(conn.W['mm0'])
    print('MBONS submatrix asymmetry: ', np.round(asymm_index, 2))
    asymm_index = nn_utils.symmetry_index(conn.W['dd0'])
    print('MBINS submatrix asymmetry: ', np.round(asymm_index, 2))
    asymm_index = nn_utils.symmetry_index(conn.W['ff0'])
    print('FB/FF submatrix asymmetry: ', np.round(asymm_index, 2))
    asymm_index = nn_utils.symmetry_index(conn.matrix)
    print('Full connectivity matrix asymmetry: ', np.round(asymm_index, 2))
    print('Averaged import:')
    asymm_index = nn_utils.symmetry_index(conn_avg.W['mm0'])
    print('MBONS submatrix asymmetry: ', np.round(asymm_index, 2))
    asymm_index = nn_utils.symmetry_index(conn_avg.W['dd0'])
    print('MBINS submatrix asymmetry: ', np.round(asymm_index, 2))
    asymm_index = nn_utils.symmetry_index(conn_avg.W['ff0'])
    print('FB/FF submatrix asymmetry: ', np.round(asymm_index, 2))
    asymm_index = nn_utils.symmetry_index(conn_avg.matrix)
    print('AVG connectivity matrix asymmetry: ', np.round(asymm_index, 2))
    
    print('')

    sparsity = nn_utils.sparsity_index(conn.W['mm0'])
    print('MBONS submatrix sparsity: ', np.round(sparsity, 2))
    sparsity = nn_utils.sparsity_index(conn.W['dd0'])
    print('MBINS submatrix sparsity: ', np.round(sparsity, 2))
    sparsity = nn_utils.sparsity_index(conn.W['ff0'])
    print('FB/FF submatrix sparsity: ', np.round(sparsity, 2))
    sparsity = nn_utils.sparsity_index(conn.matrix)
    print('Full connectivity matrix sparsity: ', np.round(sparsity, 2))
    print('Averaged import:')
    sparsity = nn_utils.sparsity_index(conn_avg.W['mm0'])
    print('MBONS submatrix sparsity: ', np.round(sparsity, 2))
    sparsity = nn_utils.sparsity_index(conn_avg.W['dd0'])
    print('MBINS submatrix sparsity: ', np.round(sparsity, 2))
    sparsity = nn_utils.sparsity_index(conn_avg.W['ff0'])
    print('FB/FF submatrix sparsity: ', np.round(sparsity, 2))
    sparsity = nn_utils.sparsity_index(conn_avg.matrix)
    print('AVG connectivity matrix sparsity: ', np.round(sparsity, 2))

    #selected compartment = 'a1'

             

