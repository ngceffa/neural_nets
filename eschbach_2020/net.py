import torch
from torch import from_numpy as from_np
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from functions_ashock import Connectivity


def setPlottingColors(fontCol, backCol):
        #global plt
        plt.rcParams.update({
            "lines.color": fontCol,
            "font.size": 16,
            "font.family": "Arial",
            "figure.figsize": [14,10],
            "patch.edgecolor": fontCol,
            "text.color": fontCol,
            "axes.facecolor": backCol,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": "lightgray",
            "axes.labelcolor": fontCol,
            "axes.labelpad": 30.0,
            "xtick.color": fontCol,
            "ytick.color": fontCol,
            "grid.color": "lightgray",
            "figure.facecolor": backCol,
            "figure.edgecolor": backCol,
            "savefig.facecolor": backCol,
            "savefig.edgecolor": backCol})

def tens2np(tens):
    return [ten.detach().numpy() for ten in tens]

def fill_us_weights(wou, mbon_names, dans_names):
    return from_np(np.vstack(\
           tens2np([torch.zeros(size=(len(mbon_names) + len(dans_names), 2)),
                    wou])))

class Network:

    def __init__(self, CM, neuron_types=["MBON", "DAN", "FB"], 
                mbon_init_rate=0, dans_init_rate=0.1, feedbacks_init_rate=0.1,
                batches=30, number_of_KCs=70, number_of_neurons=140,
                dt=0.5, coding_level_KCs=0.1, tau=5, tauw=5
                ):
        """Initialization

        Args:
            CM ([type]): [description]
            neuron_types (list, optional): [description]. Defaults to ["MBON", "DAN", "FB"].
            mbon_init_rate (int, optional): [description]. Defaults to 0.
            dans_init_rate (float, optional): [description]. Defaults to 0.1.
            feedbacks_init_rate (float, optional): [description]. Defaults to 0.1.
            batches (int, optional): [description]. Defaults to 30.
            number_of_KCs (int, optional): [description]. Defaults to 70.
            number_of_neurons (int, optional): [description]. Defaults to 140.
            dt (float, optional): [description]. Defaults to 0.5.
            coding_level_KCs (float, optional): [description]. Defaults to 0.1.
            tau (int, optional): [description]. Defaults to 5.
            tauw (int, optional): [description]. Defaults to 5.
        """
        
        # "Global" parameters
        self.neuron_types = neuron_types
        self.mbon0 = mbon_init_rate
        self.dan0 = dans_init_rate
        self.fb0 = feedbacks_init_rate
        self.batches = batches
        self.n_KCs = number_of_KCs
        self.n_neurons = number_of_neurons
        self.dt = dt
        # Model parameters
        #self.f = .1 # KC coding level
        # Maximum KC-to-MBON synaptic weight
        self.wmax = 1 / (number_of_KCs * coding_level_KCs)
        self.w_init = .5 * self.wmax
        self.tau = tau # Timing-dependent plasticity window
        self.tauw = 5 # Timescale of weight modifications

        # Static network values
        self.weights = []
        self.biases = []
        self.us_weights = []
        self.MBONbiases = []

        # Static neuron information
        self.neuron_types = neuron_types
        self.neuron_idxs = {}
        self.neuron_numbers = {}
         # from emNetworkInfo
        self.neuron_names = np.r_[CM.mbons, CM.dans, CM.feeds]
        self.posMBONs = CM.positive_mbons
        self.negMBONs = CM.negative_mbons
        self.confirmed_exc = []
        self.confirmed_inh = []
        self.DAN_MBON_indices = []
        self.inhibitory_mbons_indices = CM.inhibitory_mbons_indices
        self.excitatory_mbons_indices = CM.excitatory_mbons_indices
        self.inhibitory_feeds_indices = CM.inhibitory_feeds_indices
        self.excitatory_feeds_indices = CM.excitatory_feeds_indices

        # Lesion information
        self.neuronsToRemove = []

        # Dynamic network values
        self.rates = torch.zeros(size=(number_of_neurons, batches))
        self.KC_rate = []
        self.MBONdopamine = []
        self.lowpass_MBONdopamine = []
        self.weight_updates = []
        self.weights_KC = []
        self.US_rate = []

        # Task/input values
        self.totalTime = None
        self.KC_rates = []
        self.US_rates = []
        self.target_output = []
        self.output_valence = []
        self.reset_values_during_task = True
        self.resettimes = [60, 120]
        self.taskGenerator = None # function
        self.taskParameters = None # dict
        # Recording variables
        self.recordings = {'rates': [],
                           'output_valence': [],
                           'MBONdopamine': [],
                           'lowpass_MBONdopamine': [],
                           'KC_rates': [],
                           'lowpass_KC_rates': []
                           }
    # --------------------------------------------------------------------------
    # Weight setting (and getting) functions
    # --------------------------------------------------------------------------

    def set_network_parameters_from_original(self, train_vars):

        self.Jmm, self.Jmd, self.Jmo, self.Jdm, self.Jdd, self.Jdo, self.Jom, \
        self.Jod, self.Joo, \
        self.bm, self.bd, self.bo,\
        self.wdu, self.wou, self.wrm  = train_vars
        self.weights = from_np(np.vstack([ # post -> pre
            np.hstack(tens2np([self.Jmm, self.Jmd, self.Jmo])),
            np.hstack(tens2np([self.Jdm, self.Jdd, self.Jdo])),
            np.hstack(tens2np([self.Jom, self.Jod, self.Joo]))]))
        self.biases = from_np(np.vstack(tens2np([self.bm, self.bd, self.bo])))
        self.us_weights = fill_us_weights(self.wou, CM.mbons, CM.dans)
        self.MBONbiases = self.wrm.detach()

    def set_network_parameters(self, train_vars):
        self.weights, self.biases, self.us_weights, self.MBONbiases = train_vars
    
    # ??? useless, call -> set_network_parameters_from_original instead
    # def load_from_EM(self):
    #     self.set_network_parameters_from_original(CM.original_train_vars)
        
    def load_from_saved(self, filename):
        '''Filename = Torch save file (i.e. .p / .pt)'''
        # Load variables
        loadedVars = torch.load(filename)
        if len(loadedVars)==2:
            train_vars, self.loss = loadedVars
            self.set_network_parameters_from_original(train_vars)
        elif len(loadedVars)==3:
            train_vars, self.parameters, self.loss = loadedVars
            self.set_network_parameters(train_vars)
        elif len(loadedVars)==4:
            train_vars, self.parameters, self.loss, self.taskScripts = loadedVars
            self.set_network_parameters(train_vars)
        elif len(loadedVars)==5:
            train_vars, self.parameters, self.loss, \
            self.bestModel, self.taskScripts = loadedVars
            self.set_network_parameters(train_vars)
    
    def get_pre_post_weights(self, pres, posts):
        return self.weights[posts, :][:, pres]

    def set_pre_post_weights(self, val, pres, posts):
        for post in posts:
            for pre in pres:
                self.weights[post, pre] = val

    # --------------------------------------------------------------------------
    # Setup functions
    # --------------------------------------------------------------------------
    def setup(self):
        self.get_neuron_information()
        self.get_exc_inh_neuron_idxs()
        self.set_initial_states()

    # this works only after self.weights has been updated
    def get_neuron_information(self):
        getNeuronIndices =\
     lambda names : [i for i,n in enumerate(self.neuron_names) if n in names]
        for ntype, names in zip(self.neuron_types, [CM.mbons, CM.dans, CM.feeds]):
            # Get indices of each neuron type
            self.neuron_idxs[ntype] = getNeuronIndices(names)
            self.neuron_numbers[ntype] = len(names)
        # Get indices of DAN to MBON connections
        self.DAN_MBON_indices = torch.ones(size=self.weights.shape)
        for pre in self.neuron_idxs["DAN"]:
            for post in self.neuron_idxs["MBON"]:
                self.DAN_MBON_indices[post, pre] = 0

    def get_exc_inh_neuron_idxs(self):
        d = np.zeros(len(CM.dans), dtype=bool) # no DANs are exc or inh
        self.confirmed_exc = np.r_[self.excitatory_mbons_indices, 
                                    d,  
                                    self.excitatory_feeds_indices]
        self.confirmed_inh = np.r_[self.inhibitory_mbons_indices, 
                                    d,  
                                    self.inhibitory_feeds_indices]

    def reset(self):
        for ntype, init_val in zip(
                               self.neuron_types, [self.mbon0, self.dan0, self.fb0]):
            self.rates[self.neuron_idxs[ntype], :] = init_val
        self.lowpass_MBONdopamine = \
                        torch.zeros([self.neuron_numbers["MBON"], self.batches])
        self.lowpass_KC_rates = torch.zeros([self.n_KCs, self.batches])
    
    def set_initial_states(self):
        self.reset()
        self.set_initial_KCMBON_weights()

    def set_initial_KCMBON_weights(self):
        self.weights_KC = torch.zeros(
                                size=[self.n_neurons, self.n_KCs, self.batches])
        self.weights_KC[self.neuron_idxs["MBON"], :, :] = self.w_init

    def zero_all_weights(self):
        self.weights = torch.zeros(size=self.weights.shape)

    def initialise_recordings(self):
        self.totalTime = self.KC_rates.shape[0]
        self.recordings['rates'] = \
                torch.zeros(size=[self.totalTime, self.n_neurons, self.batches])
        self.recordings['output_valence'] = \
                            torch.zeros(size=[self.totalTime, 1, self.batches])
        self.recordings['MBONdopamine'] = \
                            torch.zeros(size=[self.totalTime, 
                                              self.neuron_numbers["MBON"], 
                                              self.batches])
        self.recordings['lowpass_MBONdopamine'] = \
                            torch.zeros(size=[self.totalTime, 
                                              self.neuron_numbers["MBON"], 
                                              self.batches])
        self.recordings['lowpass_KC_rates'] = \
                            torch.zeros(size=[self.totalTime, 
                                              self.n_KCs, 
                                              self.batches])
        self.recordings['weight_updates'] = \
                            torch.zeros(size=[self.totalTime, 
                                              self.neuron_numbers["MBON"], 
                                              self.n_KCs, 
                                              self.batches])
        self.recordings['weights_KC'] = \
                            torch.zeros(size=[self.totalTime, 
                                              self.neuron_numbers["MBON"], 
                                              self.n_KCs, 
                                              self.batches])
    
    # --------------------------------------------------------------------------
    # Task functions
    # --------------------------------------------------------------------------

    def set_task_generator(self, func, parameters, taskScript):
        self.taskGenerator = func
        self.taskParameters = parameters
        self.taskScript = taskScript

    def set_task_values(self, cs, us, target):
        self.KC_rates = torch.Tensor(cs)
        self.US_rates = torch.Tensor(us)
        self.target_output = torch.Tensor(target)
        self.initialise_recordings()

    def update_task_values(self):
        # get variables returned by task generator function
        rDict = self.taskGenerator()
        self.recordings["KC_rates"] = rDict["KC_rates"] # save KC rates
        for key in ["KC_rates", "US_rates", "target_output"]:
            if key not in rDict.keys():
                raise Exception(f'{key} missing from task input variables')
        for var in rDict.keys():
            exec(f"self.{var} = torch.Tensor(rDict['{var}'])")
        self.initialise_recordings()
        
    # --------------------------------------------------------------------------
    # Lesioning functions
    # --------------------------------------------------------------------------
    def remove_neurons(self):
        '''Remove neurons by setting their activities to zero. N
        OTE: This function must be called on every iteration t of self.run()'''
        self.rates[self.neuronsToRemove,:] = 0
        
    # --------------------------------------------------------------------------
    # Model update functions
    # --------------------------------------------------------------------------

    # Useless... ???
    def get_KCUS_rates(self,t):
        self.KC_rate = self.KC_rates[t, :, :]
        self.US_rate = self.US_rates[t, :, :]

    def update_output_valence(self):
        '''Multiply MBON firing rates by MBON biases'''
        MBON_rates = self.rates[self.neuron_idxs["MBON"], :]
        self.output_valence = self.MBONbiases.mm(MBON_rates)

    def record(self,t):
        self.recordings['rates'][t, :, :] = self.rates
        self.recordings['output_valence'][t, :, :] = self.output_valence
        self.recordings['MBONdopamine'][t, :, :] = self.MBONdopamine
        self.recordings['lowpass_MBONdopamine'][t, :, :] = \
                                                    self.lowpass_MBONdopamine
        self.recordings['lowpass_KC_rates'][t, :, :] = self.lowpass_KC_rates
        self.recordings['weight_updates'][t, :, :] = \
                             self.weight_updates[self.neuron_idxs["MBON"], :, :]
        self.recordings['weights_KC'][t, :, :, :] = \
                                 self.weights_KC[self.neuron_idxs["MBON"], :, :]
    
    # Equation 1
    def update_firing_rates(self):

        # Define variable mappings
        cs, us = self.KC_rate, self.US_rate
        
        w, b, w_us = [tens.clone() for tens in [self.weights, 
                                                    self.biases, 
                                                    self.us_weights]]
        r, w_kc = self.rates, self.weights_KC

        # Set DAN-MBON weights to zero
        w *= self.DAN_MBON_indices

        # Define inputs
        #  (w.mm(r) + b).clone() # (weights * rates) + biases
        I_internal = torch.mm(w, r) + b 
        I_KC = torch.einsum('ijb,jb->ib', (w_kc, cs)) # KC inputs
        I_US = torch.mm(w_us, us) #w_us.mm_(us) # US inputs

        # Define non-linear, positive rectification function
        nonlinear = lambda x : torch.relu(torch.tanh(x))
        
        # Return new firing rate (i.e. Equation 1 in original paper)
        self.rates = (1 - self.dt) * r + self.dt * nonlinear(I_internal + I_KC + I_US)

    # Equations 2 & 3
    def update_KCtoMBON_weights(self):
        
        # Get DAN firing rates, and MBON dopamine levels
        DAN_rates = self.rates[self.neuron_idxs["DAN"], :]
        DAN_MBON_weights = self.get_pre_post_weights(self.neuron_idxs["DAN"],
                                               self.neuron_idxs["MBON"]).clone()
        # DAN firing rates * DAN-MBON weights
        self.MBONdopamine = torch.relu(DAN_MBON_weights.mm(DAN_rates)) 
        w = self.weights_KC[self.neuron_idxs["MBON"], :] # KC-MBON_weights
        
        # Multiply MBON dopamine with KC firing rates
        dopaRateMult = \
                    lambda dopa, rate : torch.einsum('ib,jb->ijb', (dopa, rate))
        stdp_update = -dopaRateMult(self.MBONdopamine, self.lowpass_KC_rates) \
                    + dopaRateMult(self.lowpass_MBONdopamine, self.KC_rate)
        # dopamine update # stdp_update += (.00001 
        # * self.lowpass_MBONdopamine).unsqueeze(1).expand(-1,n_KCs,-1)
        # next_w = w + self.dt*(stdp_update) - torch.relu(w-wmax)
         #update that does not exceed wmax
        next_w = torch.relu(w + self.dt * (stdp_update 
                                     - torch.relu(stdp_update 
                                     - (self.wmax - w))))
        self.weight_updates = (self.dt / self.tauw) * (next_w - w) 
        # current change is equal to current difference 
        # between previous (w) and next w (next_w)          
        w += self.weight_updates
        self.weights_KC[self.neuron_idxs["MBON"], :, :] = w

        # Update low-pass filter variables
        lowpass = lambda low, orig : (1 - self.dt / self.tau) * low \
                                     + (self.dt / self.tau) * orig
        self.lowpass_MBONdopamine = \
                           lowpass(self.lowpass_MBONdopamine, self.MBONdopamine)
        self.lowpass_KC_rates = lowpass(self.lowpass_KC_rates, self.KC_rate)

        # !!!!!!! This dopamine cost term was in original scripts,
        #  but is missing from this model!
        # Update cost
        # self.dacost += torch.sum(torch.pow(torch.relu(
        # DAN_rates*baselineinds_d[ti,:,:]-dinit),2))

    # -------------------------------------------------------------------------- 
    # Run model
    # --------------------------------------------------------------------------
    def run(self):
        for t in range(self.totalTime):
            self.get_KCUS_rates(t)
            self.update_firing_rates()
            self.remove_neurons()
            self.update_KCtoMBON_weights()
            self.update_output_valence()
            self.record(t)
            if self.reset_values_during_task==True:
                if t in self.resettimes:
                    self.reset()

    # --------------------------------------------------------------------------
    # Analysis
    # --------------------------------------------------------------------------

    def get_DAN_valences(self):
        MBONbiases = self.MBONbiases
        DANweights = self.get_pre_post_weights(neuronIndices["DAN"],
                                               neuronIndices["MBON"])
                                               #  nn.MBONbiases
        DAN_valences = torch.einsum('im,md->d', 
                                   (MBONbiases, DANweights)).detach().numpy()
        return DAN_valences
    
    # --------------------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------------------
    def plotBatches(self, batches=30, wait=True):
        fig, ax = plt.subplots(3, 2, figsize=(8, 10))
        # _,indices = np.where(np.sum(self.target_output, axis=0)!=0)
        # batch = np.random.choice(indices)
        # xTimes = np.linspace(0,int(self.totalTime*dt),self.totalTime)
        xTimes = np.linspace(0, self.totalTime, self.totalTime)
        getData = lambda data: data[:, :, batch].clone().detach().numpy()
        for batch in range(batches): #indices:
            for i, j in product(range(3), range(2)):
                ax[i, j].cla()
            plt.suptitle(f"Batch {batch}")
            ax[0, 0].plot(xTimes, getData(self.KC_rates))
            ax[0, 0].plot(xTimes, getData(self.US_rates)[:,0], 
                        label="US", color="red")
            ax[0, 0].plot(xTimes, -getData(self.US_rates)[:,1], 
                        color="red")
            ax[1, 0].plot(xTimes, getData(self.target_output), 
                        label="Target output", color="black")
            ax[1, 0].plot(xTimes, getData(self.recordings["output_valence"]), 
                        label="Output", color="red")
            # *self.MBONbiases.clone().detach().numpy())
            ax[2, 0].plot(xTimes, getData(self.recordings["MBONdopamine"]))
            for i, ntype in enumerate(self.neuron_types):
                ax[i, 1].imshow(getData(
                     self.recordings["rates"][:, self.neuron_idxs[ntype], :]).T,
                     aspect="auto")
            # ax[3,0].imshow(getData(self.recordings["rates"]).T, aspect="auto")
            for i in range(len(ax)):
                if i < 2:
                    ax[i, 0].set_ylim([-1.05, 1.05])
                    ax[i, 0].legend()
                ax[i, 0].set_xlim([0, self.totalTime])
            plt.draw()
            plt.pause(1)
            if wait==True:
                input('Press any button to go to next batch.')
        
if __name__=="__main__":

    pc_path = '/Users/ngc/Documents/GitHub/'
    path = pc_path + '/atg_code/mb_modeling/eschbach2020/refactored_code/'
    file_path = path + 'data/'

    # 1- connectivity matrix
    CM = Connectivity(file_path, cutoff_hemisphere_connections=2, 
                                 cutoff_homologous_connections=9) 
    # 2- make network
    nn = Network(CM)
    # nn.loadFromEM()
    nn.load_from_saved(path+"debugging_data/trainedModel.pt")
    nn.setup()
    # # Define task variables
    KC_rates, US_rates, target_output = \
           np.load(path + "debugging_data/taskInputVariables.npy", 
                   allow_pickle=True)
    nn.set_task_values(KC_rates, US_rates, target_output)

    # Eventually define neurons to lesion
    # nn.neuronsToRemove = nn.neuron_idxs["MBON"]

    # Run model
    nn.run()

    # Plot MBON activity
    _, indices = np.where(np.sum(target_output, axis=0) != 0)
    batch = 2 # indices[10]
    savedActivity = np.load(path + "debugging_data/confirmedMBONRates.npy")
    dataToPlot = nn.recordings['rates'][:, nn.neuron_idxs["MBON"], batch]
    fig,ax = plt.subplots(4, 1)
    ax[0].plot(np.max(KC_rates[:, :, batch].detach().numpy(), axis=1))
    ax[0].plot(np.max(US_rates[:, :, batch].detach().numpy(), axis=1))
    ax[1].plot(savedActivity)
    ax[2].plot(dataToPlot)
    dataToPlot = nn.recordings['output_valence'][:, 0, batch]
    ax[3].plot(dataToPlot.detach().numpy())
    plt.show()