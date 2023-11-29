#from DurableConsumptionModel_NrM import DurableConsumptionModelClass_NrM
from DurableConsumptionModel_NrM_NoUncertainty import DurableConsumptionModelClass_NrM
# from DurableConsumptionModel import DurableConsumptionModelClass
import time
# import numpy as np

import numba as nb
nb.set_num_threads(8)

T = 5
tic = time.time()

#model_negm = DurableConsumptionModelClass_NrM(name='example_negm',par={'solmethod':'negm','T':T,'do_print':True})
model_negm = DurableConsumptionModelClass_NrM(name='example_negm',par={'solmethod':'negm','T':T,'do_print':True})
model_negm.precompile_numba() # solve with very coarse grids
model_negm.solve()
model_negm.simulate()
model_negm.save()


# Plot
model_negm.decision_functions()