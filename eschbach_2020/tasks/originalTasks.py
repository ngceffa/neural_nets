import sys
sys.path.append("..")
from BaseNetwork import dt, base_path, batches, n_KCs

# Initialise classical task creation
T = 160 # set max time
cond = "control"
cwd = f"{base_path}/../originalScripts/Fig_7_code"
exec(open(f"{cwd}/loadmat.py").read())
exec(open(f"{cwd}/classical.py").read()) # get gentrials()

class originalTasks():
    def __init__(self, custom_parameters={}):
        self.ttypes = ["secondorder","extinction"]
        self.returnedTaskVariables = ["KC_rates", "US_rates", "target_output", "usValences"]
        # Set custom parameters
        for key in custom_parameters: 
            if hasattr(self, key):
                setattr(self, key, custom_parameters[key])
        self.resettimes = [60, 120]
        self.parameters = self.__dict__.copy()
        self.xTimes = np.arange(0,T*dt,dt)
    def getTrials(self):
        KC_rates = np.zeros((T,n_KCs,batches))
        US_rates = np.zeros((T,2,batches))
        target_output = np.zeros((T,1,batches))
        for b in range(batches):
            cur_ttype = np.random.choice(self.ttypes)
            s0,u0,rtarg = gentrials(ttype=cur_ttype, B=batches)
            KC_rates[:,:,b] = s0[:,:,0]
            US_rates[:,:,b] = u0[:,:,0]
            target_output[:,:,b] = rtarg[:,:,0]
        return KC_rates, US_rates, target_output
    def taskGenerator(self, runType="training"):
        # Get task variables
        KC_rates, US_rates, target_output = self.getTrials()
        # Get US valences
        usValences = np.max(US_rates,axis=0)
        usValences = [np.where(usValences.T[i,:]==0)[0][0] for i in range(batches)]
        # Return task variables
        returnedDict = {}
        for variable in self.returnedTaskVariables:
            returnedDict[variable] = eval(variable)
        return returnedDict

if __name__=="__main__":
    from BaseNetwork import *
    nn = Network()
    nn.loadFromEM()
    nn.setup()
    parameters = {}
    # parameters = {"ttypes": ["firstorder","secondorder"]}
    t = originalTasks(parameters)
    nn.setTaskGenerator(t.taskGenerator, parameters, None)
    nn.updateTaskValues()
    nn.run()
    nn.plotBatches()



