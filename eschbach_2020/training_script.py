import sys
import torch
import numpy as np
import matplotlib.pylab as plt
from train_net import TrainingNetwork
from datetime import datetime
import traceConditioning as trace


if __name__=="__main__":

    pc_path = '/Users/ngc/Documents/GitHub/'
    path = pc_path + '/atg_code/mb_modeling/eschbach2020/refactored_code/'
    file_path = path + 'data/'
    sys.path.append(f"{path}/tasks")

    # Run with step or smooth function?
    step_function = False

    # Get network index # WHAT IS THIS?
    args = sys.argv
    netCond = "control"
    getFlagArgument = lambda flag : args[args.index(flag)+1]
    try:
        networkIndex = int(getFlagArgument('-networkIndex')) # 1-20
    except:
        print("Taking default t network index value of 1.")
        networkIndex = 1 

    # Get task (requires taskGenerator, resettimes, and parameters)
    taskType = "traceConditioning"
    exec(f"from {taskType} import *")
    if step_function == False:
        t = traceConditioning({
            "cs_isi_range": [-40,40],
            "us_timing_range": [15,20],
            "t_waiting": 30
        })
        stepString = ""
    else:
        t = traceConditioning({"stepFunction": True, "cs_isi_range": [-40,40]})
        stepString = "step-"
    parameters = t.parameters
    taskScript = open(f"{base_path}/tasks/{taskType}.py").read()