import sys
import torch
import numpy as np
from argparse import Namespace
import matplotlib.pylab as plt
sys.path.append('../')
from BaseNetwork import *
from traceConditioning import traceConditioning

class delayLearning(traceConditioning):
    '''This class, extending traceConditioning, takes trials from that class and repeats the
    training pairings a given numbers of times. Everything else is kept the same.'''
    
    # Init function
    def __init__(self, custom_parameters={}): # Set negative ISIs only}
        super().__init__({
            "cs_isi_range": [-10,-3],
            "stepFunction": True,
            "t_waiting": 10,
        })
        self.repeats = 3
        self.initialTestingLength = 20
        # Set custom parameters
        for key in custom_parameters: 
            if hasattr(self, key):
                setattr(self,key, custom_parameters[key])
        self.baseTaskGenerator = super().taskGenerator
        # Set reset times
        self.old_resettimes = self.resettimes
        self.resettimes = (self.old_resettimes*self.repeats) + self.initialTestingLength
    
    # Main task generator function
    def taskGenerator(self):
        # Get original data
        originalData = self.baseTaskGenerator("testing")
        # Create repeats of task input / target output data
        repeatData = dict()
        for dtype in ['KC_rates', 'US_rates', 'target_output']:
            # Get raw data
            cur_data = originalData[dtype]
            T,N,B = cur_data.shape
            cur_training = cur_data[:self.old_resettimes,:,:]
            # Initialise current data store
            repeatData[dtype] = np.zeros(shape=(0,N,B))
            # Set initial CS
            if dtype=="KC_rates":
                kcs = np.max(cur_data,axis=0)
                cur_testing = np.zeros(shape=(self.initialTestingLength,N,B))
                csStart = int(self.initialTestingLength/2)-int(np.floor(self.cslen/2))
                cur_testing[csStart:csStart+self.cslen,:,:] = kcs
            else:
                cur_testing = np.zeros((self.initialTestingLength,N,B))
            # # Add initial testing
            if dtype!="target_output":
                repeatData[dtype] = np.concatenate((repeatData[dtype], cur_testing), axis=0)
            else:
                repeatData[dtype] = np.concatenate((repeatData[dtype], np.zeros(cur_testing.shape)), axis=0)
            # Add repeated training
            for r in range(self.repeats+1):
                # Add data
                repeatData[dtype] = np.concatenate((repeatData[dtype], cur_training), axis=0)
            # # Add final testing
            # repeatData[dtype] = np.concatenate((repeatData[dtype], cur_testing), axis=0)
        # Add additional parameters
        for dtype in ["ISIs", "usValences", "targetValences"]:
            repeatData[dtype] = originalData[dtype]
        # Set target output
        repeatData["target_output"][:,0,:] = repeatData["US_rates"][:,0,:] - repeatData["US_rates"][:,1,:]
        # Remove last US on 100% of trials
        for b in range(batches):
            if True:#np.random.choice([0,1])==0:
                repeatData["US_rates"][self.resettimes:,:,b] = 0
        # Remove first target output
        repeatData["target_output"][:self.old_resettimes+20,:,:] = 0
        # Return task variables
        return repeatData

# Testing code
if __name__=="__main__":

    from BaseNetwork import *
    nn = Network()
    nn.loadFromEM()
    nn.setup()
    parameters = {"t_waiting": 10}
    t = delayLearning(parameters)
    nn.setTaskGenerator(t.taskGenerator, parameters, None)
    nn.updateTaskValues()
    nn.run()
    nn.plotBatches()

    # # Get delayLearning data
    # t = delayLearning()

    # for i in range(2):
    #     repeatData = t.taskGenerator()

    #     # Plot
    #     nbatches = 10
    #     setPlottingColors("black","white")
    #     kc = np.max(repeatData["KC_rates"],axis=1)
    #     us = repeatData["US_rates"][:,0,:] - repeatData["US_rates"][:,1,:] # np.max(repeatData["US_rates"],axis=1)
    #     targ = repeatData["target_output"]
    #     fig,ax = plt.subplots(nbatches,1)
    #     for batch in range(nbatches):
    #         ax[batch].plot(targ[:,0,batch],color="black")        
    #         ax[batch].plot(us[:,batch])
    #         ax[batch].plot(kc[:,batch])
    #         ax[batch].plot([t.resettimes,t.resettimes],[0,1])
    #         ax[batch].set_ylim([-1,1])
    #         ax[batch].axis(False)
    #     plt.show()
