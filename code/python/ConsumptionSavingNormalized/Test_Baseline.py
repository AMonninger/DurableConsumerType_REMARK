### Import
# import os
# path="C:/Users/adria/OneDrive - Johns Hopkins/Github/AMonninger/ConsumptionSavingNotebook_Replication/code"
# os.chdir(path)

import numpy as np
from copy import deepcopy
from DurableModel_Baseline_Jeppe import DurableConsumerType, decision_function
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType, ConsumerSolution, init_idiosyncratic_shocks
import time # to time

# load the DurableConsumptionModel module
import sys
sys.path.insert(0, './ConsumptionSavingOrig')
sys.path.insert(0, './ConsumptionSavingNormalized')

import numba as nb
nb.set_num_threads(8)

T = 5
from HARK.utilities import plot_funcs_der, plot_funcs
import matplotlib.pyplot as plt
from ConsumptionSavingNormalized.DurableConsumptionModel_NrM import DurableConsumptionModelClass_NrM
from consav.grids import nonlinspace

init_durable = dict(
    init_idiosyncratic_shocks,
    **{
        "DiscFac": [0.965,0.965,0.965,0.965],
        "Rfree": [1.03,1.03,1.03,1.03],
        # Durable Specific Parameter
        "alpha": 0.9, # Cobb-Douglas parameter for non-durable good consumption in utility function
        "dDepr": 0.15, # Depreciation Rate of Durable Stock
        "adjC": 0.10, # Adjustment costs
        "d_ubar": 1e-2, # Minimum durable stock for utility function
        # For Grids
        "nNrmMin": 0.0,
        "nNrmMax": 3.0,
        "nNrmCount": 50,
        "mNrmMin": 0.0,
        "mNrmMax": 10,
        "mNrmCount": 100,
        "xNrmMin": 0.0,
        "xNrmMax": 13, # xMax = mNrmMax + (1 - adjC)* nNrmMax
        "xNrmCount": 100,
        "aNrmMin": 0.0,
        "aNrmMax": 11,  #xNrmMax+1.0
        "aNrmCount": 100,
        ### No income shocks
        "PermGroFac": [1.0, 1.0, 1.0, 1.0],
        "PermShkCount": 5,
        "PermShkStd": [0.1, 0.1, 0.1, 0.1],
        "TranShkStd": [0.1, 0.1, 0.1, 0.1],
        "TranShkCount": 5,
        "UnempPrb":  [0.0, 0.0, 0.0, 0.0],
        "IncUnemp": [0.0, 0.0, 0.0, 0.0],
        "UnempPrbRet": [0.0, 0.0, 0.0, 0.0],
        "LivPrb": [1.0, 1.0, 1.0, 1.0],
        ### Others
        "BoroCnstArt": 0,
        "BoroCnstdNrm": 0, # Borrowing Constraint of durable goods.
        "cycles": 1,
        "T_cycle": 4,
        "tol": 1e-08,
        # To construct grids differently
        "NestFac": 3,
        "grid_type": 'nonlinear',
    }
)

### Solve and time
start = time.time()
DurableReplication = DurableConsumerType(**init_durable)
DurableReplication.update_income_process()
DurableReplication.solve()#(verbose=False)
end = time.time()
time_HARK = end - start
print("time used: ", time_HARK, " seconds")