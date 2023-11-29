"""
Tests Normalized version of Jeppe's code
"""

from DurableConsumptionModel_NrM import DurableConsumptionModelClass_NrM
import time

import numba as nb
nb.set_num_threads(8)

T = 2
tic = time.time()

model_negm = DurableConsumptionModelClass_NrM(name='example_negm',
                                              par={'solmethod':'negm',
                                                   'T':T,
                                                   'alpha': 0.5,
                                                   'tau': 1.0,
                                                   'delta': 1.0,
                                                   'gamma': 1.0,
                                                   'do_print':True})
model_negm.precompile_numba() # solve with very coarse grids
model_negm.solve()
model_negm.simulate()
model_negm.save()


# Plot
model_negm.decision_functions()