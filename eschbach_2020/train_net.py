import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pylab as plt
from tqdm import tqdm
from net import Network

# Debugging on
torch.autograd.set_detect_anomaly(True)

def enforce_sign(data, sign, indices):
    if sign=='pos':
        return torch.relu(data[:,indices])
    elif sign=='neg':
        return -torch.relu(-data[:,indices])

class TrainingNetwork(Network):
    # --------------------------------------------------------------------------
    # Initialisation
    # --------------------------------------------------------------------------
    def __init__(self, CM, epochs=1500, do_plot=False, 
                network_script_name='py', training_network_script_name='py'):
        super().__init__(CM)
        # Parameters
        self.epochs = epochs
        self.lr = 2 * np.logspace(-3, 3, epochs)
        self.alphada = .1
        self.dacost = torch.zeros(1)
        self.allowedConnections = None
        self.allowedUSWeights = None
        self.doplot = do_plot
        # Stores
        self.loss, self.loss_err, self.loss_da = torch.zeros(3)
        self.recordedVariables = ["loss","loss_err","loss_da"]
        for key in self.recordedVariables:
            self.recordings[key] = [np.float('nan') for x in range(epochs)]
        self.bestLoss = torch.Tensor([float("Inf")])
        self.bestModel = []
        # Initialise plotting figure
        if do_plot==True:
            plt.figure(figsize=(10, 10))
        self.network_script = open(f"{network_script_name}.py").read()
        self.training_network_script = \
                      open(f"{training_network_script_name}.py").read()

    # --------------------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------------------
    def setup(self):
        super().setup()
        self.set_training_vars()
        self.set_optimizer()
        self.get_allowed_connections()

    def set_training_vars(self):
        self.weights.requires_grad = True
        self.biases.requires_grad = True
        self.us_weights.requires_grad = True
        self.MBONbiases.requires_grad = True
        self.train_vars = [self.weights, self.biases, 
                           self.us_weights, self.MBONbiases]

    def get_allowed_connections(self):
        weights = self.weights.clone().detach().numpy()
        us_weights = self.us_weights.clone().detach().numpy()
        self.allowedConnections = np.array(weights != 0, dtype=int)
        self.allowedUSWeights = np.array(us_weights != 0, dtype=int)

    def set_optimizer(self):
        self.opt = torch.optim.RMSprop(self.train_vars, lr=self.lr[0])

    def save_model(self, modelfname):
        torch.save((self.train_vars, self.taskParameters, 
                    self.recordings["loss"], self.bestModel,
                    [self.network_script, self.training_network_script, 
                    self.taskScript]), 
                    modelfname)
    
    # --------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------

    def run(self, epochIndex):
        # Set learning rate for each epoch
        for group in self.opt.param_groups:
            group['lr'] = self.lr[epochIndex]
        self.set_initial_states()
        super().run()
        # ensure that rates.require_grad = False after run
        self.rates = self.rates.detach() 

    def calculate_loss(self):
        r, rtarg = self.recordings["output_valence"], self.target_output
        self.loss_err = torch.sum(torch.pow(r - rtarg, 2)) / batches 
        # regularisation term for DAN activity
        self.loss_da = self.alphada * self.dacost / batches
         # total loss (i.e. error + regularisation)
        self.loss = self.loss_err + self.loss_da
        # self.rates_loss = torch.mean(self.recordings["rates"])
        # self.rates_loss += self.rates_loss

    def perform_optimisation_step(self):
        self.loss.backward() # compute gradients
        self.opt.step() # perform single optimisation step
        self.opt.zero_grad() # clear gradients

    def perform_optimisation(self):
        self.calculate_loss()
        self.perform_optimisation_step()

    def update_recordings(self, epochIndex):
        for key in self.recordedVariables:
            self.recordings[key][epochIndex] = \
                              getattr(self, key).clone().detach().numpy().item()

    def perform_training(self):
        for epochIndex in tqdm(range(self.Nepochs)):
            # Load task information
            self.update_task_values()
            # Run training step
            self.run(epochIndex)
            self.perform_optimisation()
            self.enforce_constraints()
            self.update_recordings(epochIndex)
            # Update saved model if loss is best
            if self.loss < self.bestLoss:
                self.bestLoss = self.loss.clone().detach()
                self.bestModel = []
                for var in self.train_vars:
                    self.bestModel.append(var.clone().detach())
            # Plot current results
            if self.doplot==True:
                self.plotCurrentTrainingStep(epochIndex)

    def enforce_constraints(self):
        # no gradients with respect to these operations
        with torch.no_grad(): 
            # Enforce sparse connectivty (weights>0 only if originally not zero)
            self.weights *= torch.Tensor(self.allowedConnections)
            # Enforce US connections only where they originally existed
            self.us_weights *= torch.Tensor(self.allowedUSWeights)
            # Enforce excitatory / inhibitory neurons
            self.weights[:,self.confirmed_exc] = \
                           enforce_sign(self.weights, 'pos', self.confirmed_exc)
            self.weights[:,self.confirmed_inh] = \
                           enforce_sign(self.weights, 'neg', self.confirmed_inh)
            # Enforce MBON valences
            self.MBONbiases[:,self.posMBONs] = \
                             enforce_sign(self.MBONbiases, 'pos', self.posMBONs)
            self.MBONbiases[:,self.negMBONs] = \
                             enforce_sign(self.MBONbiases, 'neg', self.negMBONs)

    # --------------------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------------------
    def plotCurrentTrainingStep(self, epochIndex):
        plt.clf()

        # Plot loss
        plt.subplot(4, 1, 1)
        plt.semilogy(self.recordings["loss"], color="black")
        plt.xlim(0, epochIndex + 15)
        plt.xlabel('Training epoch')
        plt.ylabel('Loss')

        # Get batch of interest (i.e. where target output is not 0)
        _, indices = np.where(np.sum(self.target_output.detach().numpy(), 
                              axis=0)!=0)
        batchOfInterest = indices[0]

        # Plot task variables
        plt.subplot(4, 2, 3)
        plt.plot(np.max(self.KC_rates[:, :, batchOfInterest].detach().numpy(),
                 axis=1), label="CS", color="blue")
        plt.plot(self.US_rates[:,0,batchOfInterest].detach().numpy(), 
                 label="US", color="red")
        plt.plot(-self.US_rates[:,1,batchOfInterest].detach().numpy(), 
                 color="red")
        plt.ylim(-1.05, 1.05)
        plt.xlim(0, self.totalTime)
        plt.legend()
        plt.ylabel("Activation")
        plt.xlabel("Time")

        # Plot example network outputs
        plt.subplot(4, 2, 5)
        plt.plot(np.max(
                self.target_output[:, :, batchOfInterest].detach().numpy(),
                axis=1), label="Target", color="black")
        plt.plot(np.max(self.recordings["output_valence"][:, :, batchOfInterest]\
                        .detach().numpy(),
                        axis=1), label="Output", color="red")
        plt.ylim(-1.05, 1.05)
        plt.legend()
        plt.xlim(0, self.totalTime)
        plt.ylabel("Valence")
        plt.xlabel("Time")

        # Plot example network outputs
        plt.subplot(4, 2, 7)
        plt.imshow(
            self.recordings["rates"][:, :, batchOfInterest].detach().numpy().T, 
            aspect="auto")
        plt.xlim(0, self.totalTime)
        plt.ylabel(f"All neurons")
        plt.xlabel("Time")

        # Plot firing rates
        for i,ntype in zip([4,6,8], self.neuron_types):
            plt.subplot(4, 2, i)
            dataToPlot = self.recordings['rates']\
                        [:, self.neuron_idxs[ntype], batchOfInterest]\
                        .clone().detach().numpy().T
            plt.imshow(dataToPlot, aspect="auto")
            plt.xlim(0, self.totalTime)
            plt.ylabel(f"{ntype}s")
            plt.xlabel("Time")

        # Draw
        plt.tight_layout()
        plt.draw()
        plt.pause(.1)
