### Import
import numpy as np
from copy import deepcopy
from DurableModel_Baseline import DurableConsumerType, decision_function
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

start = time.time()
model_negm_NrM = DurableConsumptionModelClass_NrM(name='example_negm_NrM',par={'solmethod':'negm','T':T,'do_print':True})
model_negm_NrM.precompile_numba() # solve with very coarse grids
model_negm_NrM.solve()

end = time.time()
time_NrM = end - start
print("time used: ", time_NrM, " seconds")