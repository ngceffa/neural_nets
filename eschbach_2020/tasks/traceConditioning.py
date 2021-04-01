import sys
import torch
import numpy as np
from argparse import Namespace
import matplotlib.pylab as plt
sys.path.append('../')
from BaseNetwork import *

class traceConditioning:
    # -----------------------------------
    # Initialisation
    # -----------------------------------
    def __init__(self, custom_parameters={}):
        # Base parameters
        self.t_waiting = 40
        self.us_timing_range = [5,10]
        self.cs_isi_range = [-40,40]
        self.train_to_test_interval = [15,25]
        self.sineFreqPower = .4
        self.gausWidth = 50
        self.stepFunction = False
        self.cslen = 3
        self.uslen = 3
        self.dt = dt
        self.returnedTaskVariables = \
            ["KC_rates", "US_rates", "target_output", \
            "ISIs", "usValences", "targetValences"]
        # Time parameters
        self.T = 0
        self.max_testing_T = 0
        self.resettimes = []
        # Set custom parameters
        for key in custom_parameters: 
            if hasattr(self, key):
                setattr(self,key, custom_parameters[key])
        # Calculate task times
        self.max_testing_T = int(self.t_waiting + np.max(self.us_timing_range) + np.max(self.cs_isi_range) + self.t_waiting + self.cslen)
        self.T = int(self.max_testing_T + np.max(self.train_to_test_interval) + self.t_waiting)
        self.resettimes = self.max_testing_T
        # Convert times to indices
        self.parameters = self.__dict__.copy()
        for att in ["T", "max_testing_T", "t_waiting", "us_timing_range", "cs_isi_range", "cslen", "uslen", "resettimes", "train_to_test_interval"]:
            setattr(self, att, self.times2Ind(getattr(self, att)))
        self.xTimes = np.arange(0,self.T*dt,dt)


    # -----------------------------------
    # Utility functions
    # -----------------------------------

    # Convert times to indices
    def times2Ind(self, times):
        '''Convert between times (seconds) and indices'''
        if type(times)==int:
            return int(times/dt)
        elif len(times)>1:
            return [int(x) for x in np.array(times, dtype=int)/dt]

    def initTaskValues(self):
        KC_rates = np.zeros([self.T,n_KCs,batches], dtype=np.float32) # KC inputs (time, cell_index)
        US_rates = np.zeros([self.T,2,batches], dtype=np.float32) # US inputs (time, valence)
        target_output = np.zeros([self.T,1,batches], dtype=np.float32) # target (i.e. correct) output (time, correct valence)
        ISIs = np.zeros(batches)
        targetValences = np.zeros(batches)
        return KC_rates,US_rates,target_output,ISIs,targetValences

    def getKCInputIndices(self):
        inds = np.random.choice(n_KCs,int(f*n_KCs),replace=False)
        stim = np.zeros(n_KCs); stim[inds] = 1
        return stim

    # -----------------------------------
    # Task functions
    # -----------------------------------
    def valenceTrace(self, isi):
        # Return step function
        step = lambda x : -(np.heaviside(x, 0)*2)+1
        if self.stepFunction==True:
            return step(isi)
        # Else return curved valence trace
        isi -= .5 # zero valence only when isi == .5
        sineWave = np.sin(np.abs(isi/2)**self.sineFreqPower)
        gaussian = np.exp(-((np.abs(isi/2)**2)/self.gausWidth))
        result = sineWave * gaussian
        if isi>=0:
            return -result
        else:
            return result

    def showValenceTrace(self):
        x = np.linspace(-40,40,num=100)
        y = [self.valenceTrace(i) for i in x]
        plt.plot(x,y,color="black")
        plt.plot([0,0],[-1,1])
        plt.show()

    # Task generator
    def taskGenerator(self, runType="training"):

        # Initialise task variables
        KC_rates, US_rates, target_output, ISIs, targetValences = self.initTaskValues()

        # Get valences (50-50%)
        usValences = np.hstack([i*np.ones(int(batches/2), dtype=int) for i in range(2)])
        np.random.shuffle(usValences)

        # Loop over batches
        for b in range(batches):

            # Set batch settings
            if runType=="training":
                deliverUS = np.random.randint(2) # Deliver US?
                repeatCS = np.random.choice([True,False]) # Repeat CS?
            elif runType=="testing":
                deliverUS = 1
                repeatCS = True
            # us_valence = np.random.randint(2) # either 0 or 1 (positive or negative)
            # usValences[b] = us_valence

            # Set US
            cur_us_time = self.t_waiting + np.random.randint(self.us_timing_range[0],self.us_timing_range[1]+1)
            US_rates[cur_us_time:(cur_us_time+self.uslen),usValences[b],b] = deliverUS # note that deliverUS is sometimes 0

            # Set first CS (A)
            cur_ISI = int(np.random.randint(self.cs_isi_range[0],self.cs_isi_range[1]+1))
            csA_time = cur_us_time + cur_ISI
            csA_KCidxs = self.getKCInputIndices()
            KC_rates[csA_time:(csA_time+self.cslen), :, b] = csA_KCidxs
            ISIs[b] = cur_ISI*dt
            
            # Set second CS (B)
            cur_interval = np.random.randint(self.train_to_test_interval[0],self.train_to_test_interval[1]+1)
            csB_time = self.max_testing_T + cur_interval
            (cur_us_time + cur_ISI + self.cslen) + cur_interval
            if repeatCS==True:
                csB_KCidxs = csA_KCidxs
            else:
                csB_KCidxs = self.getKCInputIndices()
            KC_rates[csB_time:(csB_time+self.cslen), :, b] = csB_KCidxs

            # Set target output valence
            if (deliverUS==1 and repeatCS==True):
                valenceMapping = -1 if usValences[b] == 1 else 1
                targetValence = valenceMapping * self.valenceTrace(cur_ISI)
                target_output[csB_time:(csB_time+self.cslen), :, b] = targetValence
                targetValences[b] = targetValence

        # Return task variables
        returnedDict = {}
        for variable in self.returnedTaskVariables:
            returnedDict[variable] = eval(variable)

        return returnedDict

# Main
if __name__=="__main__":

    from BaseNetwork import *
    nn = Network()
    nn.loadFromEM()
    nn.setup()
    parameters = {'cs_isi_range': [-3,-3]}
    t = traceConditioning(parameters)
    nn.setTaskGenerator(lambda : t.taskGenerator(runType="testing"), parameters, None)
    nn.updateTaskValues()
    nn.run()
    nn.plotBatches()

    # # Initialise trace conditioning trial (with custom parameters)
    # t = traceConditioning({'cs_isi_range': [-40,40]})

    # # Plot valence trace
    # t.showValenceTrace()

    # # Get task variables
    # rDict = t.taskGenerator()
    # for var in t.returnedTaskVariables:
    #     exec(f"{var} = rDict['{var}']")

    # # Show trials
    # plt.figure()
    # for b in range(batches):
    #     plt.plot(t.xTimes,KC_rates[:,:,b])
    #     plt.plot(t.xTimes,US_rates[:,0,b], color="black")
    #     plt.plot(t.xTimes,-US_rates[:,1,b], color="black")
    #     plt.plot(t.xTimes,target_output[:,:,b], color="black")
    #     plt.plot([t.max_testing_T*dt,t.max_testing_T*dt],[-1,1])
    #     plt.ylim([-1.1,1.1])
    #     print(f"ISI = {ISIs[b]}")
    #     plt.draw()
    #     plt.pause(.1)
    #     input("Next batch?")
    #     plt.clf()