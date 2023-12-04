"""
This file contains classes and functions for representing, solving, and simulating
a consumer type with idiosyncratic shocks to permanent and transitory income,
who can consume durable and non-durable goods. There is an adjustment cost for
adjusting the durable stock as well as a depreciation rate.

Author:
Adrian Monninger

Using Code from Tianyang He: https://github.com/econ-ark/HARK/pull/825/files
and Jeppe Druedahl: https://github.com/NumEconCopenhagen/ConsumptionSavingNotebooks

Structure is based on ConsRiskyContribModel from Mateo

########################################################################################################################
Improvements:
1. Adjuster problem in last period: analytical solution instead of optimisation problem
2. Adjustment is evaluated at array instead of function
3. Inverse Value Function is the actual inverse and not -1/vFunc
4. mNrmGrid is endogeneously created
5. vFunc is similar to IndShockConsumer if the latter is produced with a spline interpolation instead of a cubic one
6. Consumption Functions uses LowerEnvelope with a Constrained and Unconstrained function.
    - Constrained Part is similar to Terminal Solution
7. Natural Borrowing Constrained is established and the aXtraGrid is similar to IndShockConsumer
8. Renamed uPFunc to vPFunc
9. Fixed vPFunc: DID I?
10. Postdecision Functions work with expected
11. Solve the Keeper with Lower-Envelope. Solve the Adjuster over inv_vFunc over Keeper Solution function (Not only unconstraint part)
12. Adjust aNrmGrid such that there is no kink at last grid point (m_next is now larger than m_treshold of following period)
TODO:
- Natural Borrowing Constrained with Durables
- With alpha = 0.5 cFunc != dFunc!

USES ROOTFINDING AND FUES INSTEAD OF JEPPE's NESTED EGM
"""
import numpy as np
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType, init_idiosyncratic_shocks
from HARK.econforgeinterp import LinearFast, DecayInterp
from HARK.interpolation import LowerEnvelope, LowerEnvelope2D, ValueFuncCRRA, BilinearInterp

from HARK.distribution import (
    combine_indep_dstns,
    DiscreteDistribution,
    IndexDistribution,
    MeanOneLogNormal,
    add_discrete_outcome_constant_mean,
)
# Additional:
from HARK.core import MetricObject, NullFunc # Basic HARK features
from copy import deepcopy
from HARK.ConsumptionSaving.ConsIndShockModel import utility, utility_inv, utilityP_inv, LognormPermIncShk, MixtureTranIncShk
from HARK.utilities import make_grid_exp_mult, construct_assets_grid
from HARK.distribution import Uniform, expected, Lognormal
#from scipy import sparse as sp
import scipy as sp
# from scipy import sparse as sp

from scipy.optimize import fsolve, root
# import estimagic as em

from utilities_Durable import construct_grid, jump_to_grid_dur, gen_tran_matrix_dur, compute_erg_dstn, DCEGM_Latest, MargValueFuncCRRA_dur, durable_adjusting_function, FUES_EGM, Upper_Envelope_Jeppe, m_nrm_next, n_nrm_next, vp_next_vPFunc, vFunc_next, EGM_njit, UpperEnvelope_njit, optimization_problem_with_guess, durable_adjusting_function_fast, durable_solution_function_fast, durable_solution_function_FUES_fast, find_roots_in_interval
from joblib import Parallel, delayed

from FUES import FUES

def TranShkMean_Func(taxrate, labor, wage):
    z = (1 - taxrate) * labor * wage
    return z



########################################################################################################################
# Make a dictionary to specify an idiosyncratic income shocks consumer
init_durable = dict(
    init_idiosyncratic_shocks,
    **{
        "alpha": 0.9, # Cobb-Douglas parameter for non-durable good consumption in utility function
        "dDepr": 0.1, # Depreciation Rate of Durable Stock
        "adjC": 0.15, # Adjustment costs
        "d_ubar": 1e-2, # Minimum durable stock for utility function
        # For Grids
        "nNrmMin": 0.0,
        "nNrmMax": 10,
        "nNrmCount": 100,
        "mNrmMin": 0.0,
        "mNrmMax": 10,
        "mNrmCount": 100,
        "xNrmCount": 200,
        "BoroCnstdNrm": 0,  # Borrowing Constraint of durable goods.
        "tol": 1e-15, # Tolerance for optimizer/ Acceptable difference before switching from adjuster to keeper
        "nNrmInitMean": 0, # Initial mean of durable stock.
        "NestFac": 3, # To construct grids differently
        "grid_type": 'exp_mult',
        "solve_terminal": True, # Boolean for solving terminal solution
        "AdjX": True, # Creating adjuster problem with a one dimensional x Grid
        "expected_BOOL": True, # Using Function expected (currently not used)
        "UpperEnvelope": 'DCEGM', # Which UpperEnvelope function do we use? DCEGM, FUES, or JEPPE
        "BoroCnstArt":[None],
        "TranMatrixnCount": 100,
        "TranMatrixmCount": 200,
        "TranMatrixnMax": 1000,
        "TranMatrixmMax": 1000,
        "TranMatrixnMin": 0,
        "TranMatrixmMin": 1e-6,
        "Limit": False, # Boolean if Limit should be calculated or not
        "extrap_method": "decay_hark",
        "taxrate": [0.0],
        "labor": [1.0],
        "wage": [1.0],
        "TranShkMean_Func": [TranShkMean_Func],
    }
)
# ### Adding taxrate, wage, labor
# init_durable['taxrate'] = [0.0] * init_durable['T_cycle']
# init_durable['labor'] = [1.0] * init_durable['T_cycle']
# init_durable['wage'] = [1.0] * init_durable['T_cycle']

class DurableConsumerType_Latest(IndShockConsumerType):
    time_inv_ = IndShockConsumerType.time_inv_ + [
        "alpha",
        "dDepr",
        "adjC",
        "d_ubar",
        "nNrmMin",
        "nNrmMax",
        "nNrmCount",
        "mNrmMin",
        "mNrmMax",
        "mNrmCount",
        "xNrmCount",
        "BoroCnstdNrm",
        "tol",
        "nNrmInitMean",
        "NestFac",
        "grid_type",
        "solve_terminal",
        "AdjX",
        "expected_BOOL",
        "UpperEnvelope",
        "Limit",
        "extrap_method",

    ]
    # Allow Rfree, Discount Rate, and Artificial Borrowing Constraint to vary over time.
    time_inv_.remove("DiscFac")
    time_inv_.remove("BoroCnstArt")
    time_vary_ = IndShockConsumerType.time_vary_ + ["Rfree", "DiscFac", "BoroCnstArt", "taxrate", "wage", "labor"]
    '''
    Adding the new state variable:
    nNrm: Beginning of period durable stock normalized by permanent income
    dNrm: End of period durable stock normalized by permanent income
    '''
    state_vars = IndShockConsumerType.state_vars + [
    "nNrm",
    "dNrm",
    ]

    def __init__(self, **kwds): # verbose=1, quiet=False,
        params = init_durable.copy()
        params.update(kwds)
        # Initialize a basic consumer type
        IndShockConsumerType.__init__(self, **params) # verbose=verbose, quiet=quiet,

        self.time_inv = deepcopy(self.time_inv_)

        self.def_utility_funcs()
        # Set the solver for the durable model, and update various constructed attributes
        self.solve_one_period = solve_DurableConsumer
        self.update()

    def def_utility_funcs(self):
        # i. U(C,D)
        u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
        self.CRRAutility = lambda C, D: utility(u_inner(C, D, self.d_ubar, self.alpha), self.CRRA)

        # ii. Inverse U(C,D)
        self.CRRAutility_inv = lambda C, D: utility_inv(self.CRRAutility(C, D), self.CRRA)

        # iii. uPC U(C,D) wrt C
        self.CRRAutilityP = lambda C, D: (
                                             ((self.alpha * C ** (self.alpha * (1 - self.CRRA) - 1)) * (D + self.d_ubar) ** ((1 - self.alpha) * (1 - self.CRRA))))
        # ((self.alpha * C ** (self.alpha * (1 - self.CRRA) - 1)) * (D + self.d_ubar) ** ((1 - self.alpha) * (1 - self.CRRA)))
        # iv. Inverse uPC U(C,D) wrt C
        # self.CRRAutilityP_inv = lambda C, D: utilityP_inv(self.CRRAutilityP(C, D), self.CRRA)
        self.CRRAutilityP_inv = lambda C, D: (
                (C / (self.alpha * (D + self.d_ubar) ** ((1 - self.alpha) * (1 - self.CRRA)))) ** (1 / (self.alpha * (1 - self.CRRA) - 1)))

        # v. uPD U(C,D) wrt D
        self.CRRAutilityPD = lambda C, D: (
                    (1 - self.alpha) * (C ** (self.alpha * (1 - self.CRRA))) * (D + self.d_ubar) ** ((1 - self.alpha) * (1 - self.CRRA) - 1))

    def pre_solve(self):
        if self.solve_terminal == True:
            self.solve_terminal_solution()
            self.update_solution_terminal()

    def update(self):
        '''
        We need to initialize multiple grids:
        1. Normalized durable stock grid: nNrmGrid
        2. Normalized market resource grid: mNrmGrid
        3. Normalized market resources + durable stock grid including adjustment costs: xNrmGrid
        4. Normalized asset grid: aNrmGrid
        '''
        self.update_income_process()

        self.updatenNrmGrid()
        self.updatemNrmGrid()
        self.updatexNrmGrid()
        self.update_assets_grid() # aXtraGrid

    def update_income_process(self):
        """
        Updates this agent's income process based on his own attributes.

        Parameters
        ----------
        none

        Returns:
        -----------
        none
        """
        # Make sure they have the same length
        if len(self.taxrate) < self.T_cycle:
            self.taxrate = [self.taxrate[0]] * self.T_cycle

        if len(self.TranShkMean_Func) < self.T_cycle:
            self.TranShkMean_Func = [self.TranShkMean_Func[0]] * self.T_cycle

        if len(self.labor) < self.T_cycle:
            self.labor = [self.labor[0]] * self.T_cycle

        if len(self.wage) < self.T_cycle:
            self.wage = [self.wage[0]] * self.T_cycle

        # for var in [self.taxrate]: #, self.TranShkMean_Func, self.labor, self.wage]:
        #     if len(var) < self.T_cycle:
        #         print(var)
        #         var = [var[0]]*self.T_cycle
        (
            IncShkDstn,
            PermShkDstn,
            TranShkDstn,
        ) = self.construct_lognormal_income_process_unemployment()
        self.IncShkDstn = IncShkDstn
        self.PermShkDstn = PermShkDstn
        self.TranShkDstn = TranShkDstn
        self.add_to_time_vary("IncShkDstn", "PermShkDstn", "TranShkDstn")
        
    def updatenNrmGrid(self): #Grid of Normalized Durable Stock
        self.nNrmGrid = construct_grid(self.nNrmMin, self.nNrmMax, self.nNrmCount, self.grid_type, self.NestFac)
        self.nNrmGrid = np.sort(np.unique(np.append(0.0, self.nNrmGrid)))
        self.add_to_time_inv('nNrmGrid')
        self.nNrmGridNow = self.nNrmGrid

    def updatemNrmGrid(self): # Grid of Normalized Market resouces if d\neq n
        # self.mNrmGrid = construct_grid(self.mNrmMin, self.mNrmMax, self.mNrmCount, self.grid_type, self.NestFac)
        self.mNrmGrid = construct_grid(0.0, self.aXtraMax, self.aXtraCount, self.grid_type, self.NestFac)
        self.add_to_time_inv('mNrmGrid')

    def updatexNrmGrid(self): # x = m + (1 - Adjc) d
        self.xNrmGrid = construct_grid(self.mNrmMin + (1 - self.adjC) * self.nNrmMin,self.mNrmMax + (1 - self.adjC) * self.nNrmMax ,self.xNrmCount, self.grid_type, self.NestFac)
        self.add_to_time_inv('xNrmGrid')

    def update_assets_grid(self):
        """
        Updates this agent's end-of-period assets grid by constructing a multi-
        exponentially spaced grid of aXtra values.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        # aXtraGrid = construct_assets_grid(self)
        # Adding a linear grid (to give it more points at the end)
        # aXtraGrid_added = construct_grid(self.aXtraMin, self.aXtraMax, self.aXtraCount, "linear", self.NestFac)
        # aXtraGrid = np.append(aXtraGrid, aXtraGrid_added)
        # aXtraGrid = np.sort(aXtraGrid)
        # aXtraGrid = np.unique(aXtraGrid)

        # self.aXtraGrid = construct_grid(self.aXtraMin, self.aXtraMax, self.aXtraCount, self.grid_type, self.NestFac)
        self.aXtraGrid = construct_assets_grid(self)
        self.add_to_time_inv("aXtraGrid")

    # Solve last period
    def solve_terminal_solution(self):
        '''
        Solves the terminal period. As there is no bequest, the agent consumes all income.
        Market resources and durable stock can be split into durable and non-durable consumption.

        Keepers problem: consume every market resource (m) given durable stock (d).

        Adjuster problem: Calculate optimal share of durables which is c = alpha*xNrmGrid; d = (1-alpha)*xNrmGrid

        Optimal Solution: Use either keeper or adjuster solution depending on the higher utility.

        :return:
        cFunc : function
            The consumption function for this period, defined over durable stock and market
            resources: c = cFunc(n,m).
        dFunc: function
            The durable consumption function for this period, defined over durable stock and
            market resources: d = dFunc(n,m)
        exFunc: function
            The total expenditure function for this period, defined over durable stock and
            market resources: ex = exFunc(n,m)
        vFunc : function
            The beginning-of-period value function for this period, defined over durable stock
            and market resources: v = vFunc(n,m).
        cFuncKeep : function
            The consumption function for this periods' keeper problem, defined over durable stock and market
            resources: c = cFunc(n,m).
        dFuncKeep: function
            The durable consumption function for this periods' keeper problem, defined over durable stock and
            market resources: d = dFunc(n,m).
        exFuncKeep: function
            The total expenditure function for this periods' keeper problem, defined over durable stock and
            market resources: ex = exFunc(n,m)
        vFuncKeep : function
            The beginning-of-period value function for this periods' keeper problem, defined over durable stock
            and market resources: v = vFunc(n,m).
        cFuncAdj : function
            The consumption function for this periods' adjuster problem, defined over sum of durable stock and market
            resources: c = cFunc(x).
        dFuncAdj: function
            The durable consumption function for this periods' adjuster problem, defined over sum of durable stock and
            market resources: d = dFunc(x)
        exFuncAdj: function
            The total expenditure function for this periods' adjuster problem, defined over sum of durable stock and
            market resources: ex = exFunc(x)
        vFuncAdj : function
            The beginning-of-period value function for this periods' adjuster problem, defined over sum of durable stock
            and market resources: v = vFunc(x).
        adjusting : function
            The adjusting function for this period indicates if for a given durable stock and market resources, the agent
            adjusts the durable stock or not: adjusting = adjusting(n,m).
        '''
        # # Use nNrmGrid which we need in period T-1: # Needs to be adjusted by shock!
        # if not self.dDepr == 1:
        #     self.nNrmGridNow = self.nNrmGridNow * (1 - self.dDepr)
        # self.nNrmGridNow = np.append(self.nNrmGridNow, 1000.0)
        # a) keeper problem: keep durable stock and consume everything else
        self.cFuncKeep_terminal = LinearFast(np.array([[0.0,1.0],[0.0,1.0]]), [np.array([0.0,1.0]), np.array([0.0,1.0])])
        self.dFuncKeep_terminal = LinearFast(np.array([[0.0,0.0],[1.0,1.0]]), [np.array([0.0,1.0]), np.array([0.0,1.0])])
        self.exFuncKeep_terminal = LinearFast(np.array([[0.0,1.0],[1.0,2.0]]), [np.array([0.0,1.0]), np.array([0.0,1.0])])

        keep_shape = (len(self.nNrmGridNow),len(self.mNrmGrid))

        # # Value and marginal value Functions
        # # i) empty container
        inv_vFuncKeep_terminal_array = np.zeros(keep_shape)
        # # ii) fill arrays
        for i_m in range(len(self.mNrmGrid)):
            if self.mNrmGrid[i_m] == 0:  # forced c = 0
                inv_vFuncKeep_terminal_array[:, i_m] = 0
                continue
            m_aux = np.ones(len(self.nNrmGridNow)) * self.mNrmGrid[i_m]
            inv_vFuncKeep_terminal_array[:,i_m] = self.CRRAutility_inv(m_aux, self.nNrmGridNow)

        # iii) Make Functions
        self.inv_vFuncKeep_terminal = LinearFast(inv_vFuncKeep_terminal_array, [self.nNrmGridNow, self.mNrmGrid])
        self.vFuncKeep_terminal = ValueFuncCRRA(self.cFuncKeep_terminal, self.CRRA)
        self.vPFuncKeep_terminal = MargValueFuncCRRA_dur(self.cFuncKeep_terminal, self.dFuncKeep_terminal, self.CRRA,
                                                        self.alpha, self.d_ubar)
        # b) adjuster problem: (Note this is an analytical solution)
        cFuncAdj_terminal_array = self.alpha * self.xNrmGrid
        dFuncAdj_terminal_array = (1 - self.alpha) * self.xNrmGrid

        inv_vFuncAdj_terminal_array = self.CRRAutility_inv(self.xNrmGrid - dFuncAdj_terminal_array, dFuncAdj_terminal_array)

        # Create Functions
        self.cFuncAdj_terminal = LinearFast(cFuncAdj_terminal_array, [self.xNrmGrid])
        self.dFuncAdj_terminal = LinearFast(dFuncAdj_terminal_array, [self.xNrmGrid])
        self.exFuncAdj_terminal = LinearFast(self.xNrmGrid, [self.xNrmGrid])
        self.inv_vFuncAdj_terminal = LinearFast(inv_vFuncAdj_terminal_array, [self.xNrmGrid])
        self.vFuncAdj_terminal = ValueFuncCRRA(self.cFuncAdj_terminal, self.CRRA)
        self.vPFuncAdj_terminal = MargValueFuncCRRA_dur(self.cFuncAdj_terminal, self.dFuncAdj_terminal, self.CRRA,
                                                          self.alpha, self.d_ubar)

        # c) Create Consumption Function: Using analytical solution
        # The solution to an S,s model is adjust - keep - adjust.
        # Here, we get the upper and lower bound of the keeper region
        # Adjuster for index <= lS
        # Adjuster for ls > index < uS

        solution_shape = (len(self.nNrmGridNow), 2)
        lSuS_array = np.zeros(solution_shape)
        # Solve S,s bands analytical. Only sensible if utility from durables and adjustment costs are not zero (frictionless)
        if self.alpha < 1 and self.adjC > 0: # Only useful
            def f(m):
                return (self.alpha * (m + (1 - self.adjC) * n)) ** self.alpha * (
                            (1 - self.alpha) * (m + (1 - self.adjC) * n)) ** (
                               1 - self.alpha) - m ** self.alpha * n ** (1 - self.alpha)

            for i_n in range(len(self.nNrmGridNow)):
                if i_n == 0:
                    lSuS_array[0] = [0.0, 0.0]
                else:
                    n = self.nNrmGridNow[i_n]
                    lSuS_array[i_n] = fsolve(f, [lSuS_array[i_n-1,0], lSuS_array[i_n-1,1]+1])

        if np.any(np.logical_not(np.all(np.diff(lSuS_array, axis=0) >= 0, axis=0))):
            print("fsolve did not work properly!")

        self.mNrmGridTotal = lSuS_array.reshape((len(self.nNrmGridNow) * 2, 1))
        # Add a point to the left of the threshold
        # self.mNrmGridTotal_extra = np.abs(self.mNrmGridTotal - 0.0000000001)
        # self.mNrmGridTotal = np.sort(np.unique(np.append(self.mNrmGridTotal, self.mNrmGridTotal_extra)))
        self.mNrmGridTotal = np.sort(np.unique(self.mNrmGridTotal))
        if self.alpha < 1:
            self.mNrmGridTotal = np.append(self.mNrmGridTotal, np.max([np.max(self.mNrmGridTotal), self.mNrmMax]) + 1)
            # self.mNrmGridTotal = np.append(self.mNrmGridTotal, [np.max([np.max(self.mNrmGridTotal), self.mNrmMax]) + 1, np.max([np.max(self.mNrmGridTotal), self.mNrmMax]) +2])
        else:
            self.mNrmGridTotal = np.append(self.mNrmGridTotal, 1)

        solution_shape = (len(self.nNrmGridNow), len(self.mNrmGridTotal))
        cFunc_array = np.zeros(solution_shape)
        dFunc_array = np.zeros(solution_shape)
        inv_vFunc_array = np.zeros(solution_shape)
        vPFuncD_array = np.zeros(solution_shape)
        adjusting_array = np.ones(solution_shape)

        for i_d in range(len(self.nNrmGridNow)):
            lS_idx = np.where(self.mNrmGridTotal == lSuS_array[i_d][0])[0][0]
            uS_idx = np.where(self.mNrmGridTotal == lSuS_array[i_d][1])[0][0]
            length = np.max([0, uS_idx - lS_idx - 1])
            d_aux = np.ones(len(self.mNrmGridTotal)) * self.nNrmGridNow[i_d]
            x_aux = self.mNrmGridTotal + (1 - self.adjC)*d_aux

            cFunc_array[i_d] = self.alpha * x_aux
            cFunc_array[i_d][lS_idx + 1: uS_idx] = self.mNrmGridTotal[lS_idx + 1: uS_idx]

            dFunc_array[i_d] = (1 - self.alpha) * x_aux
            dFunc_array[i_d][lS_idx + 1: uS_idx] = d_aux[lS_idx + 1: uS_idx]

            inv_vFunc_array[i_d] = self.CRRAutility_inv(self.alpha * x_aux, (1 - self.alpha) * x_aux) #self.inv_vFuncAdj_terminal(d_aux[:lS_idx], self.mNrmGridTotal[:lS_idx])
            inv_vFunc_array[i_d][lS_idx + 1: uS_idx] = self.CRRAutility_inv(self.mNrmGridTotal[lS_idx + 1: uS_idx], d_aux[lS_idx + 1: uS_idx]) #self.inv_vFuncKeep_terminal(d_aux[lS_idx: uS_idx], self.mNrmGridTotal[lS_idx: uS_idx])

            vPFuncD_array[i_d] = (1 - self.adjC) * self.CRRAutilityP(cFunc_array[i_d], dFunc_array[i_d]) # Adjust
            vPFuncD_array[i_d][lS_idx + 1: uS_idx] = self.CRRAutilityPD(cFunc_array[i_d][lS_idx + 1: uS_idx],
                                                                            dFunc_array[i_d][lS_idx + 1: uS_idx]) # Keep

            adjusting_array[i_d, lS_idx + 1:uS_idx] = np.zeros(length)

        exFunc_array = cFunc_array + dFunc_array
        # Interpolation
        self.cFunc_terminal = LinearFast(cFunc_array, [self.nNrmGridNow, self.mNrmGridTotal])
        self.dFunc_terminal = LinearFast(dFunc_array, [self.nNrmGridNow, self.mNrmGridTotal])
        self.exFunc_terminal = LinearFast(exFunc_array, [self.nNrmGridNow, self.mNrmGridTotal])
        self.adjusting_terminal = LinearFast(adjusting_array, [self.nNrmGridNow, self.mNrmGridTotal])
        self.vFunc_terminal = ValueFuncCRRA(self.cFunc_terminal, self.CRRA)
        self.vPFunc_terminal = MargValueFuncCRRA_dur(self.cFunc_terminal, self.dFunc_terminal, self.alpha, self.d_ubar, self.CRRA)
        self.inv_vFunc_terminal = LinearFast(inv_vFunc_array, [self.nNrmGridNow, self.mNrmGridTotal])
        self.vPFuncD_terminal = LinearFast(vPFuncD_array, [self.nNrmGridNow, self.mNrmGridTotal])
        ####################################################################################################################
        if np.mean(adjusting_array) == 1:
            adjustment_only_shape = (len(self.nNrmGridNow), len(self.mNrmGrid))
            inv_vFunc_array = np.zeros(adjustment_only_shape)
            vPFuncD_array = np.zeros(adjustment_only_shape)
            for i_d in range(len(self.nNrmGridNow)):
                d_aux = np.ones(len(self.mNrmGrid)) * self.nNrmGridNow[i_d]
                x_aux = self.mNrmGrid + (1 - self.adjC) * d_aux
                inv_vFunc_array[i_d] = self.CRRAutility_inv(self.alpha * x_aux, (1 - self.alpha) * x_aux) #self.inv_vFuncAdj_terminal(d_aux[:lS_idx], self.mNrmGridTotal[:lS_idx])
                vPFuncD_array[i_d] = (1 - self.adjC) * self.CRRAutilityP(self.alpha * x_aux, (1 - self.alpha) * x_aux) # Adjust

            self.inv_vFunc_terminal = LinearFast(inv_vFunc_array, [self.nNrmGridNow, self.mNrmGrid])
            self.vPFuncD_terminal = LinearFast(vPFuncD_array, [self.nNrmGridNow, self.mNrmGrid])


        ###
        ###

        self.mNrmMin=0.0
        self.hNrm=0.0
        self.MPCmin=1.0
        self.MPCmax=1.0
        ###
    def update_solution_terminal(self):
        self.solution_terminal = DurableConsumerSolution(
            cFuncKeep = self.cFuncKeep_terminal,
            cFuncAdj = self.cFuncAdj_terminal,
            dFuncKeep = self.dFuncKeep_terminal,
            dFuncAdj = self.dFuncAdj_terminal,
            exFuncKeep = self.exFuncKeep_terminal,
            exFuncAdj = self.exFuncAdj_terminal,
            vFuncKeep = self.vFuncKeep_terminal,
            vFuncAdj = self.vFuncAdj_terminal,
            vPFuncKeep = self.vPFuncKeep_terminal,
            vPFuncAdj = self.vPFuncAdj_terminal,
            inv_vFuncKeep = self.inv_vFuncKeep_terminal,
            inv_vFuncAdj = self.inv_vFuncAdj_terminal,
            cFunc = self.cFunc_terminal,
            dFunc = self.dFunc_terminal,
            exFunc = self.exFunc_terminal,
            vFunc = self.vFunc_terminal,
            vPFunc = self.vPFunc_terminal,
            vPFuncD = self.vPFuncD_terminal,
            inv_vFunc = self.inv_vFunc_terminal,
            adjusting = self.adjusting_terminal,
            mNrmGrid=self.mNrmGrid,
            nNrmGrid=self.nNrmGrid,
            mNrmMin = self.mNrmMin,
            hNrm = self.hNrm,
            MPCmin = self.MPCmin,
            MPCmax = self.MPCmax,
            aXtraGrid=self.aXtraGrid,
            # inv_vFuncKeepUnc = None,
            # cFuncKeepUnc = None,
            # exFuncKeepUnc=None,
            # # Constraint functions
            cFuncKeepCnst = self.cFuncKeep_terminal,
            dFuncKeepCnst=self.dFuncKeep_terminal,
            exFuncKeepCnst=self.exFuncKeep_terminal,
            cFuncCnst = self.cFunc_terminal,
            dFuncCnst = self.dFunc_terminal,
            exFuncCnst = self.exFunc_terminal,
            # # m_thresh_max = self.m_thresh_max,
            # # Comparing Upper Envelope
            qFunc = None,
            qFuncD = None,
            wFunc = None,
            aNrmGridNow = None,
            # c_egm_array=None,
            # m_egm_array=None,
            # v_egm_array=None,
            BoroCnstNat = None,
    )

    def construct_lognormal_income_process_unemployment(self):
        """
        Generates a list of discrete approximations to the income process for each
        life period, from end of life to beginning of life.  Permanent shocks are mean
        one lognormally distributed with standard deviation PermShkStd[t] during the
        working life, and degenerate at 1 in the retirement period.  Transitory shocks
        are mean one lognormally distributed with a point mass at IncUnemp with
        probability UnempPrb while working; they are mean one with a point mass at
        IncUnempRet with probability UnempPrbRet.  Retirement occurs
        after t=T_retire periods of working.

        Note 1: All time in this function runs forward, from t=0 to t=T

        Note 2: All parameters are passed as attributes of the input parameters.

        Parameters (passed as attributes of the input parameters)
        ----------
        PermShkStd : [float]
            List of standard deviations in log permanent income uncertainty during
            the agent's life.
        PermShkCount : int
            The number of approximation points to be used in the discrete approxima-
            tion to the permanent income shock distribution.
        TranShkStd : [float]
            List of standard deviations in log transitory income uncertainty during
            the agent's life.
        TranShkCount : int
            The number of approximation points to be used in the discrete approxima-
            tion to the permanent income shock distribution.
        UnempPrb : float or [float]
            The probability of becoming unemployed during the working period.
        UnempPrbRet : float or None
            The probability of not receiving typical retirement income when retired.
        T_retire : int
            The index value for the final working period in the agent's life.
            If T_retire <= 0 then there is no retirement.
        IncUnemp : float or [float]
            Transitory income received when unemployed.
        IncUnempRet : float or None
            Transitory income received while "unemployed" when retired.
        T_cycle :  int
            Total number of non-terminal periods in the consumer's sequence of periods.

        Returns
        -------
        IncShkDstn :  [distribution.Distribution]
            A list with T_cycle elements, each of which is a
            discrete approximation to the income process in a period.
        PermShkDstn : [[distribution.Distributiony]]
            A list with T_cycle elements, each of which
            a discrete approximation to the permanent income shocks.
        TranShkDstn : [[distribution.Distribution]]
            A list with T_cycle elements, each of which
            a discrete approximation to the transitory income shocks.
        """
        # Unpack the parameters from the input
        T_cycle = self.T_cycle
        PermShkStd = self.PermShkStd
        PermShkCount = self.PermShkCount
        TranShkStd = self.TranShkStd
        TranShkCount = self.TranShkCount
        T_retire = self.T_retire
        UnempPrb = self.UnempPrb
        IncUnemp = self.IncUnemp
        UnempPrbRet = self.UnempPrbRet
        IncUnempRet = self.IncUnempRet

        # Make sure they have the same length
        if len(self.taxrate) < self.T_cycle:
            self.taxrate = [self.taxrate[0]] * self.T_cycle

        if len(self.TranShkMean_Func) < self.T_cycle:
            self.TranShkMean_Func = [self.TranShkMean_Func[0]] * self.T_cycle

        if len(self.labor) < self.T_cycle:
            self.labor = [self.labor[0]] * self.T_cycle

        if len(self.wage) < self.T_cycle:
            self.wage = [self.wage[0]] * self.T_cycle

        # for var in [self.taxrate]: #, self.TranShkMean_Func, self.labor, self.wage]:
        #     if len(var) < self.T_cycle:
        #         print(var)
        #         var = [var[0]]*self.T_cycle

        taxrate = self.taxrate
        TranShkMean_Func = self.TranShkMean_Func
        labor = self.labor
        wage = self.wage

        if T_retire > 0:
            normal_length = T_retire
            retire_length = T_cycle - T_retire
        else:
            normal_length = T_cycle
            retire_length = 0

        if all(
                [
                    isinstance(x, (float, int)) or (x is None)
                    for x in [UnempPrb, IncUnemp, UnempPrbRet, IncUnempRet]
                ]
        ):

            UnempPrb_list = [UnempPrb] * normal_length + [UnempPrbRet] * retire_length
            IncUnemp_list = [IncUnemp] * normal_length + [IncUnempRet] * retire_length

        elif all([isinstance(x, list) for x in [UnempPrb, IncUnemp]]):

            UnempPrb_list = UnempPrb
            IncUnemp_list = IncUnemp

        else:

            raise Exception(
                "Unemployment must be specified either using floats for UnempPrb,"
                + "IncUnemp, UnempPrbRet, and IncUnempRet, in which case the "
                + "unemployment probability and income change only with retirement, or "
                + "using lists of length T_cycle for UnempPrb and IncUnemp, specifying "
                + "each feature at every age."
            )

        PermShkCount_list = [PermShkCount] * normal_length + [1] * retire_length
        TranShkCount_list = [TranShkCount] * normal_length + [1] * retire_length

        if not hasattr(self, "neutral_measure"):
            self.neutral_measure = False

        neutral_measure_list = [self.neutral_measure] * len(PermShkCount_list)

        '''
        IncShkDstn = IndexDistribution(
            engine=BufferStockIncShkDstn,
            conditional={
                "sigma_Perm": PermShkStd,
                "sigma_Tran": TranShkStd,
                "n_approx_Perm": PermShkCount_list,
                "n_approx_Tran": TranShkCount_list,
                "neutral_measure": neutral_measure_list,
                "UnempPrb": UnempPrb_list,
                "IncUnemp": IncUnemp_list,
            },
            RNG=self.RNG,
        )

        '''
        IncShkDstn = IndexDistribution(
            engine=HANKIncShkDstn,
            conditional={
                "sigma_Perm": PermShkStd,
                "sigma_Tran": TranShkStd,
                "n_approx_Perm": PermShkCount_list,
                "n_approx_Tran": TranShkCount_list,
                "neutral_measure": neutral_measure_list,
                "UnempPrb": UnempPrb_list,
                "IncUnemp": IncUnemp_list,
                "wage": wage,
                "taxrate": taxrate,
                "labor": labor,
                "TranShkMean_Func": TranShkMean_Func,
            },
            RNG=self.RNG,
        )

        PermShkDstn = IndexDistribution(
            engine=LognormPermIncShk,
            conditional={
                "sigma": PermShkStd,
                "n_approx": PermShkCount_list,
                "neutral_measure": neutral_measure_list,
            },
        )
        '''
        TranShkDstn = IndexDistribution(
            engine=MixtureTranIncShk,
            conditional={
                "sigma": TranShkStd,
                "UnempPrb": UnempPrb_list,
                "IncUnemp": IncUnemp_list,
                "n_approx": TranShkCount_list,
            },
        )
        '''

        TranShkDstn = IndexDistribution(
            engine=MixtureTranIncShk_HANK,
            conditional={
                "sigma": TranShkStd,
                "UnempPrb": UnempPrb_list,
                "IncUnemp": IncUnemp_list,
                "n_approx": TranShkCount_list,
                "wage": wage,
                "taxrate": taxrate,
                "labor": labor,
                "TranShkMean_Func": TranShkMean_Func,

            },
        )

        return IncShkDstn, PermShkDstn, TranShkDstn

    ####################################################################################################################
    ### SIMULATION PART STARTS HERE: Only functions are shown which are changed
    def sim_birth(self, which_agents):
        """
        Makes new consumers for the given indices.  Initialized variables include aNrm and pLvl, as
        well as time variables t_age and t_cycle.  Normalized assets and permanent income levels
        are drawn from lognormal distributions given by aNrmInitMean and aNrmInitStd (etc).

        Additionally, creates nNrm (normalized durable stock) drawn from lognormal distribution.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        IndShockConsumerType.sim_birth(self, which_agents)
        # Adding nNrm for birth
        N = np.sum(which_agents)  # Number of new consumers to make
        # self.state_now["nNrm"][which_agents] = Lognormal(
        self.state_now["dNrm"][which_agents] = (Lognormal(
            mu=self.nNrmInitMean,
            sigma=self.nNrmInitStd,
            seed=self.RNG.randint(0, 2**31 - 1),
        ).draw(N))/(1 - self.dDepr) #* self.shocks['PermShk']
        if self.dDepr < 1:
            self.state_now["dNrm"][which_agents] = self.state_now["dNrm"][which_agents]/(1 - self.dDepr)
        else:
            self.state_now["dNrm"][which_agents] = 0
        
        # As this is End-of-Period durable stock, divide by depreciation and shocks such that beginning of period durable
        # stock is equal to draw.
        return None

    def transition(self): # Adding nNrm
        pLvlPrev = self.state_prev["pLvl"]
        aNrmPrev = self.state_prev["aNrm"]
        dNrmPrev = self.state_prev["dNrm"]
        RfreeNow = self.get_Rfree()

        # Calculate new states: normalized market resources and permanent income level
        # Updated permanent income level
        pLvlNow = pLvlPrev * self.shocks["PermShk"]
        # Updated aggregate permanent productivity level
        PlvlAggNow = self.state_prev["PlvlAgg"] * self.PermShkAggNow
        # "Effective" interest factor on normalized assets
        ReffNow = RfreeNow / self.shocks["PermShk"]
        bNrmNow = ReffNow * aNrmPrev  # Bank balances before labor income
        # Market resources after income
        mNrmNow = bNrmNow + self.shocks["TranShk"]

        nNrmNow = dNrmPrev * (1 - self.dDepr)/self.shocks['PermShk'] # Normalized Durable Stock
        return pLvlNow, PlvlAggNow, bNrmNow, mNrmNow, None, None, nNrmNow, None
        # Nones are aNrm, aLvl, dNrm as those are defined by the choices made in the period.

    def get_controls(self):
        """
        Calculates consumption for each consumer of this type using the consumption functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        # Added
        dNrmNow = np.zeros(self.AgentCount) + np.nan
        exNrmNow = np.zeros(self.AgentCount) + np.nan
        # adjusting = np.zeros(self.AgentCount) + np.nan

        MPCnow = np.zeros(self.AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            # CHANGE: cFunc has dimensions (nNrm, mNrm) Optimal c given the durable stock nNrm and market resources
            cNrmNow[these] = self.solution[t].cFunc(self.state_now['nNrm'][these], self.state_now['mNrm'][these])
            dNrmNow[these] = self.solution[t].dFunc(self.state_now['nNrm'][these], self.state_now['mNrm'][these])
            exNrmNow[these] = self.solution[t].exFunc(self.state_now['nNrm'][these], self.state_now['mNrm'][these])
            # adjusting[these] = self.solution[t].adjusting(self.state_now['nNrm'][these], self.state_now['mNrm'][these])

            # cNrmNow[these], MPCnow[these] = self.solution[t].cFunc.eval_with_derivative(
            #     self.state_now["mNrm"][these]
            # )
        self.controls['cNrm'] = cNrmNow
        self.controls['dNrm'] = dNrmNow
        self.controls['exNrm'] = exNrmNow
        # self.controls['adjusting'] = adjusting
        return None

        # # MPCnow is not really a control
        # self.MPCnow = MPCnow
        # return None

    def get_poststates(self):
        """
        Calculates end-of-period assets for each consumer of this type.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Poststates depend on decision:
        self.state_now['aNrm'] = np.zeros(self.AgentCount)
        self.state_now['adjusting'] = np.zeros(self.AgentCount)
        xNrm = self.state_now['mNrm'] + (1 - self.adjC) * self.state_now['nNrm']
        # for i_Agent in range(self.AgentCount):
        #     if self.AdjX == False and self.solution[self.t_cycle[i_Agent]].inv_vFuncAdj(self.state_now['nNrm'][i_Agent], self.state_now['mNrm'][i_Agent]) + self.tol >= self.solution[self.t_cycle[i_Agent]].inv_vFuncKeep(self.state_now['nNrm'][i_Agent], self.state_now['mNrm'][i_Agent]):
        #         # if self.controls['adjusting'][i_Agent]:
        #         self.state_now['aNrm'][i_Agent] = xNrm[i_Agent] - self.controls['cNrm'][i_Agent] - self.controls['dNrm'][i_Agent]
        #         self.state_now['adjusting'][i_Agent] = 1
        #     elif self.AdjX == True and self.solution[self.t_cycle[i_Agent]].inv_vFuncAdj(xNrm[i_Agent]) + self.tol >= self.solution[self.t_cycle[i_Agent]].inv_vFuncKeep(self.state_now['nNrm'][i_Agent], self.state_now['mNrm'][i_Agent]):
        #         self.state_now['aNrm'][i_Agent] = xNrm[i_Agent] - self.controls['cNrm'][i_Agent] - self.controls['dNrm'][
        #             i_Agent]
        #         self.state_now['adjusting'][i_Agent] = 1
        #     else:
        #         self.state_now['aNrm'][i_Agent] = self.state_now['mNrm'][i_Agent] - self.controls['cNrm'][i_Agent]

        ### TEST: don't loop over each agent
        ### Create Boolean vector for all agents with True: adjusting and False: not adjusting
        ###
        if self.AdjX == False:
            adjusting_Bool = self.solution[self.t_cycle[0]].inv_vFuncAdj(self.state_now['nNrm'], self.state_now['mNrm']) + self.tol >= self.solution[self.t_cycle[0]].inv_vFuncKeep(self.state_now['nNrm'], self.state_now['mNrm'])
        else:
            adjusting_Bool = self.solution[self.t_cycle[0]].inv_vFuncAdj(xNrm) + self.tol >= self.solution[self.t_cycle[0]].inv_vFuncKeep(self.state_now['nNrm'], self.state_now['mNrm'])

        self.state_now['aNrm'][adjusting_Bool] = xNrm[adjusting_Bool] - self.controls['cNrm'][adjusting_Bool] - self.controls['dNrm'][
                    adjusting_Bool]

        self.state_now['aNrm'][~adjusting_Bool] = self.state_now['mNrm'][~adjusting_Bool] - self.controls['cNrm'][~adjusting_Bool]
        self.state_now['adjusting'][adjusting_Bool] = np.ones(self.AgentCount)[adjusting_Bool]

        # Useful in some cases to precalculate asset level
        self.state_now['aLvl'] = self.state_now['aNrm'] * self.state_now['pLvl']

        # Add durable stocks normalized and in levels
        self.state_now['dNrm'] = self.controls['dNrm'] #Durable consumption this period is equal to the stock next period
        self.state_now['dLvl'] = self.state_now['dNrm'] * self.state_now['pLvl']
        self.state_now['nLvl'] = self.state_now['nNrm'] * self.state_now['pLvl']
        # moves now to prev
        # super().get_poststates() # If this is true, then calculates poststates based on ConsIndShockModel

        return None

    ### TRANSITION MATRIX:
    def define_distribution_grid_dur(self, dist_mGrid=None, dist_nGrid=None, m_density=0, timestonest = None):

        """
        Defines the grid on which the distribution is defined. Stores the grid of market resources and permanent income as attributes of self.
        Grid for normalized market resources and durable stock may be prespecified
        as dist_mGrid and dist_nGrid, respectively. If not then default grid is computed based off given parameters.

        Added: durable grid

        Parameters
        ----------
        dist_mGrid : np.array
                Prespecified grid for distribution over normalized market resources

        dist_nGrid : np.array
                 Prespecified grid for distribution over normalized durable stock.

        m_density: float
                Density of normalized market resources grid. Default value is mdensity = 0.
                Only affects grid of market resources if dist_mGrid=None.

        TranMatrixmCount: float
                Number of gridpoints for market resources grid.

        TranMatrixnCount: float
                Number of gridpoints for durable stock grid.

        Returns
        -------
        None
        """

        if not hasattr(self,
                       "neutral_measure"):  # If true Use Harmenberg 2021's Neutral Measure. For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch
            self.neutral_measure = False

        # if TranMatrixmCount == None:
        #     TranMatrixmCount = self.aXtraCount
        # else:
        #     m_points = num_pointsM
        #
        # if num_pointsN == None:
        #     n_points = self.nNrmCount
        # else:
        #     n_points = num_pointsN

        if not isinstance(timestonest, int):
            timestonest = self.aXtraNestFac
        else:
            timestonest = timestonest

        if self.cycles == 0:

            if not hasattr(dist_mGrid, '__len__'):

                aXtra_Grid = make_grid_exp_mult(
                    ming=self.TranMatrixmMin, maxg=self.TranMatrixmMax, ng=self.TranMatrixmCount,
                    timestonest=timestonest)  # Generate Market resources grid given density and number of points

                for i in range(m_density):
                    axtra_shifted = np.delete(aXtra_Grid, -1)
                    axtra_shifted = np.insert(axtra_shifted, 0, 1.00000000e-04)
                    dist_betw_pts = aXtra_Grid - axtra_shifted
                    dist_betw_pts_half = dist_betw_pts / 2
                    new_A_grid = axtra_shifted + dist_betw_pts_half
                    aXtra_Grid = np.concatenate((aXtra_Grid, new_A_grid))
                    aXtra_Grid = np.sort(aXtra_Grid)

                self.dist_mGrid = aXtra_Grid

            else:
                self.dist_mGrid = dist_mGrid  # If grid of market resources prespecified then use as mgrid

            if not hasattr(dist_nGrid, '__len__'):

                nXtra_Grid = construct_grid(self.TranMatrixnMin, self.TranMatrixnMax, self.TranMatrixnCount, self.grid_type, self.NestFac)
                for i in range(m_density):
                    nxtra_shifted = np.delete(nXtra_Grid, -1)
                    nxtra_shifted = np.insert(nxtra_shifted, 0, 1.00000000e-04)
                    dist_betw_pts = nXtra_Grid - nxtra_shifted
                    dist_betw_pts_half = dist_betw_pts / 2
                    new_N_grid = nxtra_shifted + dist_betw_pts_half
                    nXtra_Grid = np.concatenate((nXtra_Grid, new_N_grid))
                    nXtra_Grid = np.sort(nXtra_Grid)
                self.dist_nGrid = nXtra_Grid
            else:
                self.dist_nGrid = dist_nGrid  # If grid of durable stock prespecified then use it as ngrid

        elif self.cycles > 1:
            raise Exception('define_distribution_grid requires cycles = 0 or cycles = 1')

        elif self.T_cycle != 0:
            # if num_pointsM == None:
            #     m_points = self.aXtraCount
            # else:
            #     m_points = num_pointsM

            if not hasattr(dist_mGrid, '__len__'):
                aXtra_Grid = make_grid_exp_mult(
                    ming=self.TranMatrixmMin, maxg=self.TranMatrixmMax, ng=self.TranMatrixmCount,
                    timestonest=timestonest)  # Generate Market resources grid given density and number of points

                for i in range(m_density):
                    axtra_shifted = np.delete(aXtra_Grid, -1)
                    axtra_shifted = np.insert(axtra_shifted, 0, 1.00000000e-04)
                    dist_betw_pts = aXtra_Grid - axtra_shifted
                    dist_betw_pts_half = dist_betw_pts / 2
                    new_A_grid = axtra_shifted + dist_betw_pts_half
                    aXtra_Grid = np.concatenate((aXtra_Grid, new_A_grid))
                    aXtra_Grid = np.sort(aXtra_Grid)

                self.dist_mGrid = aXtra_Grid

            else:
                self.dist_mGrid = dist_mGrid  # If grid of market resources prespecified then use as mgrid

            if not hasattr(dist_nGrid, '__len__'):

                nXtra_Grid = construct_grid(self.TranMatrixnMin, self.TranMatrixnMax, self.TranMatrixnCount, self.grid_type, self.NestFac)
                for i in range(m_density):
                    nxtra_shifted = np.delete(nXtra_Grid, -1)
                    nxtra_shifted = np.insert(nxtra_shifted, 0, 1.00000000e-04)
                    dist_betw_pts = nXtra_Grid - nxtra_shifted
                    dist_betw_pts_half = dist_betw_pts / 2
                    new_N_grid = nxtra_shifted + dist_betw_pts_half
                    nXtra_Grid = np.concatenate((nXtra_Grid, new_N_grid))
                    nXtra_Grid = np.sort(nXtra_Grid)
                self.dist_nGrid = nXtra_Grid
            else:
                self.dist_nGrid = dist_nGrid  # If grid of durable stock prespecified then use it as ngrid

    # 3. Calculate transition matrix
    def calc_transition_matrix_dur(self, shk_dstn=None):
        '''
        Calculates how the distribution of agents across market resources
        transitions from one period to the next. If finite horizon problem, then calculates
        a list of transition matrices, consumption and asset policy grids for each period of the problem.
        The transition matrix/matrices and consumption and asset policy grid(s) are stored as attributes of self.


        Parameters
        ----------
            shk_dstn: list
                list of income shock distributions
        Returns
        -------
        None

        '''

        if self.cycles == 0:

            if not hasattr(shk_dstn, 'pmv'):
                shk_dstn = self.IncShkDstn

            dist_nGrid = self.dist_nGrid  # Grid of durable stock
            dist_mGrid = self.dist_mGrid  # Grid of market resources

            # get aNext which depends on decision
            aNext_shape = (len(dist_nGrid), len(dist_mGrid))
            aNext = np.zeros(aNext_shape)
            self.cPol_Grid = np.zeros(aNext_shape)
            self.dPol_Grid = np.zeros(aNext_shape)
            self.exPol_Grid = np.zeros(aNext_shape)

            for i_n in range(len(dist_nGrid)):
                n_aux = np.ones(len(dist_mGrid)) * dist_nGrid[i_n]
                x_aux = dist_mGrid + (1 - self.adjC) * n_aux

                self.cPol_Grid[i_n] = self.solution[0].cFunc(n_aux, dist_mGrid)  # Steady State Consumption Policy Grid
                self.dPol_Grid[i_n] = self.solution[0].dFunc(n_aux, dist_mGrid)
                self.exPol_Grid[i_n] = self.solution[0].exFunc(n_aux, dist_mGrid)

                # Depending on adjusting or not, we have a different budget constraint

                if self.AdjX == False:
                    adjusting_Bool = self.solution[0].inv_vFuncAdj(n_aux, dist_mGrid) + self.tol >= \
                                 self.solution[0].inv_vFuncKeep(n_aux, dist_mGrid)
                else:
                    adjusting_Bool = self.solution[0].inv_vFuncAdj(x_aux) + self.tol >= \
                                     self.solution[0].inv_vFuncKeep(n_aux, dist_mGrid)

                aNext[i_n][adjusting_Bool] = x_aux[adjusting_Bool] - self.exPol_Grid[i_n][adjusting_Bool]
                aNext[i_n][~adjusting_Bool] = dist_mGrid[~adjusting_Bool] - self.cPol_Grid[i_n][~adjusting_Bool]


            self.aPol_Grid = aNext  # Steady State Asset Policy Grid


            # ### OLD
            # for i_n in range(len(dist_nGrid)):
            #     for i_m in range(len(dist_mGrid)):
            #         self.cPol_Grid[i_n, i_m] = self.solution[0].cFunc(dist_nGrid[i_n],
            #                                                 dist_mGrid[i_m])  # Steady State Consumption Policy Grid
            #         self.dPol_Grid[i_n, i_m] = self.solution[0].dFunc(dist_nGrid[i_n], dist_mGrid[i_m])
            #         self.exPol_Grid[i_n, i_m] = self.solution[0].exFunc(dist_nGrid[i_n], dist_mGrid[i_m])
            #
            #         # Depending on adjusting or not, we have a different budget constraint
            #         if self.solution[0].adjusting(dist_nGrid[i_n], dist_mGrid[i_m]) < 1:
            #             aNext[i_n, i_m] = dist_mGrid[i_m] - self.solution[0].cFunc(dist_nGrid[i_n],
            #                                                                   dist_mGrid[i_m])
            #             if aNext[i_n,i_m] < 0:
            #                 aNext[i_n, i_m] = 0
            #         else:
            #             aNext[i_n, i_m] = dist_mGrid[i_m] + (1 - self.adjC) * dist_nGrid[i_n] - self.exPol_Grid[i_n, i_m]
            #             if aNext[i_n,i_m] < 0:
            #                 aNext[i_n, i_m] = 0
            #
            # self.aPol_Grid_Old = aNext  # Steady State Asset Policy Grid
            #
            # ### Vectorized
            # ### VECTORIZED
            #
            # dist_nGrid = self.dist_nGrid  # Grid of durable stock
            # dist_mGrid = self.dist_mGrid  # Grid of market resources
            #
            # # get aNext which depends on decision
            # n_aux = np.outer(dist_nGrid, np.ones(len(dist_mGrid)))
            # x_aux = dist_mGrid + (1 - self.adjC) * n_aux
            #
            # self.cPol_Grid = self.solution[0].cFunc(n_aux, dist_mGrid)  # Steady State Consumption Policy Grid
            # self.dPol_Grid = self.solution[0].dFunc(n_aux, dist_mGrid)
            # self.exPol_Grid = self.solution[0].exFunc(n_aux, dist_mGrid)
            #
            # # Depending on adjusting or not, we have a different budget constraint
            # if self.AdjX == False:
            #     adjusting_Bool = self.solution[0].inv_vFuncAdj(n_aux, dist_mGrid) + self.tol >= \
            #                      self.solution[0].inv_vFuncKeep(n_aux, dist_mGrid)
            # else:
            #     adjusting_Bool = self.solution[0].inv_vFuncAdj(x_aux) + self.tol >= \
            #                      self.solution[0].inv_vFuncKeep(n_aux, dist_mGrid)
            #
            # aNext = np.where(adjusting_Bool, x_aux - self.exPol_Grid, dist_mGrid - self.cPol_Grid)
            #
            # self.aPol_Grid_Vec = aNext  # Steady State Asset Policy Grid




            # Obtain shock values and shock probabilities from income distribution
            bNext = self.Rfree[0] * aNext  # Bank Balances next period (Interest rate * assets)
            #bNext = self.Rfree * aNext  # Bank Balances next period (Interest rate * assets)
            nNext_unshocked = (1 - self.dDepr) * self.dPol_Grid  # Durable Stock next period after depreciation
            shk_prbs = shk_dstn[0].pmv #shk_dstn[0].pmf  # Probability of shocks
            tran_shks = shk_dstn[0].atoms[1] #shk_dstn[0].X[1]  # Transitory shocks
            perm_shks = shk_dstn[0].atoms[0] #shk_dstn[0].X[0]  # Permanent shocks
            LivPrb = self.LivPrb[0]  # Update probability of staying alive

            # New borns have this distribution (assumes start with no assets, no durable stock and permanent income=1)
            NewBornDist = jump_to_grid_dur(np.zeros_like(tran_shks), tran_shks,  shk_prbs, dist_nGrid, dist_mGrid)

            # Generate Transition Matrix
            self.tran_matrix = gen_tran_matrix_dur(dist_nGrid, dist_mGrid, bNext, nNext_unshocked, shk_prbs, perm_shks, tran_shks, LivPrb,
                                NewBornDist)

        elif self.cycles > 1:
            print('calc_transition_matrix requires cycles = 0 or cycles = 1')

        elif self.T_cycle != 0: # finite horizon problem

            if not hasattr(shk_dstn, 'pmv'):
                shk_dstn = self.IncShkDstn

            self.cPol_Grid = []  # List of non-durable consumption policy grids for each period in T_cycle
            self.dPol_Grid = [] # List of durable consumption policy grids for each period in T_cycle
            self.exPol_Grid = []  # List of total expenditure consumption policy grids for each period in T_cycle
            self.aPol_Grid = []  # List of asset policy grids for each period in T_cycle
            self.tran_matrix = []  # List of transition matrices

            dist_mGrid = self.dist_mGrid # Grid of market resources

            for k in range(self.T_cycle):

                # If grid is time varying
                if type(self.dist_nGrid) == list:
                    dist_nGrid = self.dist_nGrid[k]  # Durable stock grid this period
                else:
                    dist_nGrid = self.dist_nGrid  # If here then use prespecified durable stock grid

                aNext_shape = (len(dist_nGrid), len(dist_mGrid))
                Cnow = np.zeros(aNext_shape)
                Dnow = np.zeros(aNext_shape)
                Exnow = np.zeros(aNext_shape)
                aNext = np.zeros(aNext_shape)

                for i_n in range(len(dist_nGrid)):
                    n_aux = np.ones(len(dist_mGrid)) * dist_nGrid[i_n]
                    x_aux = dist_mGrid + (1 - self.adjC) * n_aux

                    Cnow[i_n] = self.solution[k].cFunc(n_aux,
                                                                 dist_mGrid)  # Steady State Consumption Policy Grid
                    Dnow[i_n] = self.solution[k].dFunc(n_aux, dist_mGrid)
                    Exnow[i_n] = self.solution[k].exFunc(n_aux, dist_mGrid)

                    # Depending on adjusting or not, we have a different budget constraint
                    # If adjust: x - c - d
                    # If keep: m - c

                    if self.AdjX == False:
                        adjusting_Bool = self.solution[k].inv_vFuncAdj(n_aux, dist_mGrid) + self.tol >= \
                                         self.solution[k].inv_vFuncKeep(n_aux, dist_mGrid)
                    else:
                        adjusting_Bool = self.solution[k].inv_vFuncAdj(x_aux) + self.tol >= \
                                         self.solution[k].inv_vFuncKeep(n_aux, dist_mGrid)

                    aNext[i_n][adjusting_Bool] = x_aux[adjusting_Bool] - Exnow[i_n][adjusting_Bool]
                    aNext[i_n][~adjusting_Bool] = dist_mGrid[~adjusting_Bool] - Cnow[i_n][~adjusting_Bool]

                ### OLD
                # for i_n in range(len(dist_nGrid)):
                #     for i_m in range(len(dist_mGrid)):
                #         Cnow[i_n, i_m] = self.solution[k].cFunc(dist_nGrid[i_n], dist_mGrid[i_m])
                #         Dnow[i_n, i_m] = self.solution[k].dFunc(dist_nGrid[i_n], dist_mGrid[i_m])
                #         Exnow[i_n, i_m] = self.solution[k].exFunc(dist_nGrid[i_n], dist_mGrid[i_m])
                #         # Depending on adjusting or not, we have a different budget constraint
                #         if self.solution[k].adjusting(dist_nGrid[i_n], dist_mGrid[i_m]) < 1:
                #             aNext[i_n, i_m] = dist_mGrid[i_m] - Cnow[i_n,i_m] #Cnow(dist_nGrid[i_n],dist_mGrid[i_m])
                #             if aNext[i_n, i_m] < 0:
                #                 aNext[i_n, i_m] = 0
                #         else:
                #             aNext[i_n, i_m] = dist_mGrid[i_m] + (1 - self.adjC) * dist_nGrid[i_n] \
                #                               - Exnow[i_n, i_m] #Exnow(dist_nGrid[i_n], dist_mGrid[i_m])
                #             if aNext[i_n, i_m] < 0:
                #                 aNext[i_n, i_m] = 0
                self.cPol_Grid.append(Cnow) # Add to list
                self.dPol_Grid.append(Dnow) # Add to list
                self.exPol_Grid.append(Exnow) # Add to list
                self.aPol_Grid.append(aNext) # Add to list

                if type(self.Rfree) == list:
                    bNext = self.Rfree[k] * aNext
                else:
                    bNext = self.Rfree * aNext

                nNext_unshocked = (1 - self.dDepr) * Dnow
                # Obtain shocks and shock probabilities from income distribution this period
                shk_prbs = shk_dstn[k].pmv #shk_dstn[k].pmf  # Probability of shocks this period
                tran_shks = shk_dstn[k].atoms[1] #shk_dstn[k].X[1]  # Transitory shocks this period
                perm_shks = shk_dstn[k].atoms[0] #shk_dstn[k].X[0]  # Permanent shocks this period
                LivPrb = self.LivPrb[k]  # Update probability of staying alive this period

                # New borns have this distribution (assumes start with no assets, no durable stock and permanent income=1)
                NewBornDist = jump_to_grid_dur(np.zeros_like(tran_shks), tran_shks, shk_prbs, dist_nGrid, dist_mGrid)

                # Generate Transition Matrix
                TranMatrix = gen_tran_matrix_dur(dist_nGrid, dist_mGrid, bNext, nNext_unshocked, shk_prbs,
                                                       perm_shks, tran_shks, LivPrb,
                                                       NewBornDist)
                # self.tran_matrix.append(TranMatrix)
                TranMatrix_sparse = sp.sparse.csr_matrix(TranMatrix)
                self.tran_matrix.append(TranMatrix_sparse)


    # 4. Calculate ergodic distribution: calc_ergodic_dist
    def calc_ergodic_dist(self, transition_matrix=None, method=None):

        '''
        Calculates the ergodic distribution across normalized market resources and
        permanent income as the eigenvector associated with the eigenvalue 1.
        The distribution is stored as attributes of self both as a vector and as a reshaped array with the ij'th element representing
        the probability of being at the i'th point on the mGrid and the j'th
        point on the pGrid.

        Parameters
        ----------
        transition_matrix: List
                    transition matrix whose ergordic distribution is to be solved
        Returns
        -------
        None
        '''

        if not isinstance(transition_matrix, list):
            transition_matrix = [self.tran_matrix]

        # ergodic_distr = compute_erg_dstn(transition_matrix[0], method)
        if method == 'Pontus':
            ergodic_distr = compute_erg_dstn(transition_matrix[0])
        else:
            eigen, ergodic_distr = sp.sparse.linalg.eigs(transition_matrix[0], v0=np.ones(len(transition_matrix[0])), k=1,
                                              which='LM')  # Solve for ergodic distribution
            ergodic_distr = ergodic_distr.real / np.sum(ergodic_distr.real)



        self.vec_erg_dstn = ergodic_distr  # distribution as a vector
        self.erg_dstn = ergodic_distr.reshape(
            (len(self.dist_nGrid), len(self.dist_mGrid)))  # distribution reshaped into len(ngrid) by len(mgrid) array

    def compute_steady_state(self):
        # Compute steady state to perturb around
        self.cycles = 0
        self.solve()

        # Use Harmenberg Measure
        self.neutral_measure = True
        self.update_income_process()

        # Non stochastic simuation
        self.define_distribution_grid_dur()
        self.calc_transition_matrix_dur()

        self.c_ss = self.cPol_Grid  # Normalized Consumption Policy grid
        self.d_ss = self.dPol_Grid  # Normalized Durable Policy grid
        self.ex_ss = self.exPol_Grid # Normalized Total Expenditure Policy grid
        self.a_ss = self.aPol_Grid  # Normalized Asset Policy grid

        self.calc_ergodic_dist()  # Calculate ergodic distribution
        # Steady State Distribution as a vector (m*p x 1) where m is the number of gridpoints on the market resources grid
        ss_dstn = self.vec_erg_dstn

        # Need to flatten policy grids.
        self.A_ss = np.dot(self.a_ss.flatten(), ss_dstn)[0]
        self.C_ss = np.dot(self.c_ss.flatten(), ss_dstn)[0]
        self.EX_ss = np.dot(self.ex_ss.flatten(), ss_dstn)[0]
        self.D_ss = np.dot(self.d_ss.flatten(), ss_dstn)[0]

        return self.A_ss, self.C_ss, self.D_ss, self.EX_ss

    def calc_jacobian(self, shk_param, T):
        """
        Calculates the Jacobians of aggregate consumption and aggregate assets. Parameters that can be shocked are
        LivPrb, PermShkStd,TranShkStd, DiscFac, UnempPrb, Rfree, IncUnemp, DiscFac .

        Parameters:
        -----------

        shk_param: string
            name of variable to be shocked

        T: int
            dimension of Jacobian Matrix. Jacobian Matrix is a TxT square Matrix


        Returns
        ----------
        DJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Durable Expenditure with respect to shk_param

        CJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Consumption with respect to shk_param

        EXJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Total Expenditure with respect to shk_param

        AJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Assets with respect to shk_param

        """

        # Set up finite Horizon dictionary
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = T  # Dimension of Jacobian Matrix

        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["DiscFac"] = params["T_cycle"] * [self.DiscFac[0]]
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["UnempPrbRet"] = params["T_cycle"] * [self.UnempPrbRet[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        params["IncUnempRet"] = params["T_cycle"] * [self.IncUnempRet[0]]
        params["BoroCnstArt"] = params["T_cycle"] * [self.BoroCnstArt[0]]

        # Create instance of a finite horizon agent
        FinHorizonAgent = DurableConsumerType_Latest(**params)
        FinHorizonAgent.cycles = 1  # required
        FinHorizonAgent.solve_terminal = False # Terminal period is steady state
        # # delete Rfree from time invariant list since it varies overtime
        # FinHorizonAgent.del_from_time_inv("Rfree")
        # # Add Rfree to time varying list to be able to introduce time varying interest rates
        # FinHorizonAgent.add_to_time_vary("Rfree")

        # Set Terminal Solution as Steady State Consumption Function
        # FinHorizonAgent.cFunc_terminal_ = deepcopy(self.solution[0].cFunc)

        # Additional
        FinHorizonAgent.solution_terminal.inv_vFunc = self.solution[0].inv_vFunc
        FinHorizonAgent.solution_terminal.vPFunc = self.solution[0].vPFunc
        FinHorizonAgent.solution_terminal.vPFuncD = self.solution[0].vPFuncD
        FinHorizonAgent.solution_terminal.cFunc = self.solution[0].cFunc
        FinHorizonAgent.solution_terminal.dFunc = self.solution[0].dFunc
        FinHorizonAgent.solution_terminal.cFuncKeepCnst = self.solution[0].cFuncKeepCnst
        FinHorizonAgent.solution_terminal.dFuncKeepCnst = self.solution[0].dFuncKeepCnst
        FinHorizonAgent.solution_terminal.exFuncKeepCnst = self.solution[0].exFuncKeepCnst
        FinHorizonAgent.solution_terminal.cFuncCnst = self.solution[0].cFuncCnst
        FinHorizonAgent.solution_terminal.dFuncCnst = self.solution[0].dFuncCnst
        FinHorizonAgent.solution_terminal.exFuncCnst = self.solution[0].exFuncCnst

        dx = 0.0001  # Size of perturbation
        # Period in which the change in the interest rate occurs (second to last period)
        i = params["T_cycle"] - 1

        FinHorizonAgent.IncShkDstn = params["T_cycle"] * [self.IncShkDstn[0]]

        # If parameter is in time invariant list then add it to time vary list
        FinHorizonAgent.del_from_time_inv(shk_param)
        FinHorizonAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                    (i) * [getattr(self, shk_param)[0]]
                    + [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            peturbed_list = (
                    (i) * [getattr(self, shk_param)]
                    + [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent faces
        setattr(FinHorizonAgent, shk_param, peturbed_list)

        # Update income process if perturbed parameter enters the income shock distribution
        FinHorizonAgent.update_income_process()

        # Solve
        FinHorizonAgent.solve()

        # Use Harmenberg Neutral Measure
        FinHorizonAgent.neutral_measure = True
        FinHorizonAgent.update_income_process()

        # Calculate Transition Matrices
        FinHorizonAgent.define_distribution_grid_dur()
        FinHorizonAgent.calc_transition_matrix_dur()

        # Normalized consumption Policy Grids across time
        c_t = FinHorizonAgent.cPol_Grid
        d_t = FinHorizonAgent.dPol_Grid
        ex_t = FinHorizonAgent.exPol_Grid
        a_t = FinHorizonAgent.aPol_Grid

        # Append steady state policy grid into list of policy grids as HARK does not provide the initial policy
        c_t.append(self.c_ss)
        d_t.append(self.d_ss)
        ex_t.append(self.ex_ss)
        a_t.append(self.a_ss)

        # Fake News Algorithm begins below ( To find fake news algorithm See page 2388 of https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434  )

        ##########
        # STEP 1 # of fake news algorithm, As in the paper for Curly Y and Curly D. Here the policies are over assets and consumption so we denote them as curly C and curly D.
        ##########
        a_ss = self.aPol_Grid  # steady state Asset Policy
        c_ss = self.cPol_Grid  # steady state Consumption Policy
        d_ss = self.dPol_Grid  # steady state Durable Expenditure Policy
        ex_ss = self.exPol_Grid  # steady state Total Expenditure Policy

        tranmat_ss = self.tran_matrix  # Steady State Transition Matrix

        # List of policies grids where households expect the shock to occur in the second to last Period
        a_t = FinHorizonAgent.aPol_Grid
        c_t = FinHorizonAgent.cPol_Grid
        d_t = FinHorizonAgent.dPol_Grid
        ex_t = FinHorizonAgent.exPol_Grid

        # add steady state policies to list as it does not get appended in calc_transition_matrix method
        a_t.append(self.a_ss)
        c_t.append(self.c_ss)
        d_t.append(self.d_ss)
        ex_t.append(self.ex_ss)

        da0_s = []  # Deviation of asset policy from steady state policy
        dc0_s = []  # Deviation of Consumption policy from steady state policy
        dd0_s = []  # Deviation of Durable Expenditure policy from steady state policy
        dex0_s = []  # Deviation of Total Expenditure policy from steady state policy

        for i in range(T):
            da0_s.append(a_t[T - i] - a_ss)
            dc0_s.append(c_t[T - i] - c_ss)
            dd0_s.append(d_t[T - i] - d_ss)
            dex0_s.append(ex_t[T - i] - ex_ss)

        da0_s = np.array(da0_s)
        dc0_s = np.array(dc0_s)
        dd0_s = np.array(dd0_s)
        dex0_s = np.array(dex0_s)

        # Steady state distribution of market resources (permanent income weighted distribution)
        Dist_ss = self.vec_erg_dstn.T[0] # Shape: ((nxm) x 1)
        dA0_s = []
        dC0_s = []
        dD0_s = []
        dEX0_s = []
        for i in range(T):
            dA0_s.append(np.dot(da0_s[i].flatten(), Dist_ss))
            dC0_s.append(np.dot(dc0_s[i].flatten(), Dist_ss))
            dD0_s.append(np.dot(dd0_s[i].flatten(), Dist_ss))
            dEX0_s.append(np.dot(dex0_s[i].flatten(), Dist_ss))

        # This is equivalent to the curly Y scalar detailed in the first step of the algorithm
        dA0_s = np.array(dA0_s)
        dC0_s = np.array(dC0_s)
        dD0_s = np.array(dD0_s)
        dEX0_s = np.array(dEX0_s)

        A_curl_s = dA0_s / dx # Shape (T,1)
        C_curl_s = dC0_s / dx
        D_curl_s = dD0_s / dx
        EX_curl_s = dEX0_s / dx

        # List of computed transition matrices for each period
        tranmat_t = FinHorizonAgent.tran_matrix
        tranmat_t.append(tranmat_ss)

        # List of change in transition matrix relative to the steady state transition matrix
        dlambda0_s = []
        for i in range(T):
            dlambda0_s.append(tranmat_t[T - i] - tranmat_ss)

        dlambda0_s = np.array(dlambda0_s)

        dDist0_s = []
        for i in range(T):
            dDist0_s.append(np.dot(dlambda0_s[i], Dist_ss))

        dDist0_s = np.array(dDist0_s) # Shape: T x (nxm)
        Dist_curl_s = dDist0_s / dx  # Curly D in the sequence space jacobian

        ########
        # STEP2 # of fake news algorithm
        ########

        # Expectation Vectors: Shape T x n x m
        exp_vecs_a = []
        exp_vecs_c = []
        exp_vecs_d = []
        exp_vecs_ex = []

        # First expectation vector is the steady state policy
        exp_vec_a = a_ss.flatten()
        exp_vec_c = c_ss.flatten()
        exp_vec_d = d_ss.flatten()
        exp_vec_ex = ex_ss.flatten()

        for i in range(T):
            exp_vecs_a.append(exp_vec_a)
            exp_vec_a = np.dot(exp_vec_a.flatten(), tranmat_ss)

            exp_vecs_c.append(exp_vec_c)
            exp_vec_c = np.dot(exp_vec_c.flatten(), tranmat_ss)

            exp_vecs_d.append(exp_vec_d)
            exp_vec_d = np.dot(exp_vec_d.flatten(), tranmat_ss)

            exp_vecs_ex.append(exp_vec_ex)
            exp_vec_ex = np.dot(exp_vec_ex.flatten(), tranmat_ss)

        # Turn expectation vectors into arrays
        exp_vecs_a = np.array(exp_vecs_a)
        exp_vecs_c = np.array(exp_vecs_c)
        exp_vecs_d = np.array(exp_vecs_d)
        exp_vecs_ex = np.array(exp_vecs_ex)

        #########
        # STEP3 # of the algorithm. In particular equation 26 of the published paper.
        #########
        # Fake news matrices
        Curl_F_A = np.zeros((T, T))  # Fake news matrix for assets
        Curl_F_C = np.zeros((T, T))  # Fake news matrix for consumption
        Curl_F_D = np.zeros((T, T))  # Fake news matrix for durable expenditure
        Curl_F_EX = np.zeros((T, T))  # Fake news matrix for total expenditure

        # First row of Fake News Matrix: Shaper ((n x m) x 1)
        Curl_F_A[0] = A_curl_s
        Curl_F_C[0] = C_curl_s
        Curl_F_D[0] = D_curl_s
        Curl_F_EX[0] = EX_curl_s

        for i in range(T - 1):
            for j in range(T):
                Curl_F_A[i + 1][j] = np.dot(exp_vecs_a[i].flatten(), Dist_curl_s[j])
                Curl_F_C[i + 1][j] = np.dot(exp_vecs_c[i].flatten(), Dist_curl_s[j])
                Curl_F_D[i + 1][j] = np.dot(exp_vecs_d[i].flatten(), Dist_curl_s[j])
                Curl_F_EX[i + 1][j] = np.dot(exp_vecs_ex[i].flatten(), Dist_curl_s[j])

        ########
        # STEP4 #  of the algorithm
        ########

        # Function to compute jacobian matrix from fake news matrix
        def J_from_F(F):
            J = F.copy()
            for t in range(1, F.shape[0]):
                J[1:, t] += J[:-1, t - 1]
            return J

        J_A = J_from_F(Curl_F_A)
        J_C = J_from_F(Curl_F_C)
        J_D = J_from_F(Curl_F_D)
        J_EX = J_from_F(Curl_F_EX)

        ########
        # Additional step due to compute Zeroth Column of the Jacobian
        ########
        
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = 2  # Dimension of Jacobian Matrix

        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["DiscFac"] = params["T_cycle"] * [self.DiscFac[0]]
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["UnempPrbRet"] = params["T_cycle"] * [self.UnempPrbRet[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        params["IncUnempRet"] = params["T_cycle"] * [self.IncUnempRet[0]]
        params["BoroCnstArt"] = params["T_cycle"] * [self.BoroCnstArt[0]]
        
        
        # Create instance of a finite horizon agent for calculation of zeroth
        ZerothColAgent = DurableConsumerType_Latest(**params)
        ZerothColAgent.cycles = 1  # required
        ZerothColAgent.solve_terminal = False  # Terminal period is steady state

        # Terminal solution
        ZerothColAgent.solution_terminal.inv_vFunc = self.solution[0].inv_vFunc
        ZerothColAgent.solution_terminal.vPFunc = self.solution[0].vPFunc
        ZerothColAgent.solution_terminal.vPFuncD = self.solution[0].vPFuncD
        ZerothColAgent.solution_terminal.cFunc = self.solution[0].cFunc
        ZerothColAgent.solution_terminal.dFunc = self.solution[0].dFunc
        ZerothColAgent.solution_terminal.cFuncKeepCnst = self.solution[0].cFuncKeepCnst
        ZerothColAgent.solution_terminal.dFuncKeepCnst = self.solution[0].dFuncKeepCnst
        ZerothColAgent.solution_terminal.exFuncKeepCnst = self.solution[0].exFuncKeepCnst
        ZerothColAgent.solution_terminal.cFuncCnst = self.solution[0].cFuncCnst
        ZerothColAgent.solution_terminal.dFuncCnst = self.solution[0].dFuncCnst
        ZerothColAgent.solution_terminal.exFuncCnst = self.solution[0].exFuncCnst

        # If parameter is in time invariant list then add it to time vary list
        ZerothColAgent.del_from_time_inv(shk_param)
        ZerothColAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            peturbed_list = (
                    [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            peturbed_list = (
                    [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent

        setattr(ZerothColAgent, shk_param, peturbed_list)  # Set attribute to agent

        # Update income process if perturbed parameter enters the income shock distribution
        ZerothColAgent.update_income_process()

        # Solve
        ZerothColAgent.solve()

        # Use Harmenberg Neutral Measure
        ZerothColAgent.neutral_measure = True
        ZerothColAgent.update_income_process()

        # Calculate Transition Matrices
        ZerothColAgent.define_distribution_grid_dur()
        ZerothColAgent.calc_transition_matrix_dur()

        tranmat_t_zeroth_col = ZerothColAgent.tran_matrix
        tranmat_t_zeroth_col_dense = sp.sparse.csr_matrix.toarray(tranmat_t_zeroth_col[0]) # Convert from sparse matrix to array
        dstn_t_zeroth_col = self.vec_erg_dstn.T[0]

        A_t_no_sim = np.zeros(T)
        C_t_no_sim = np.zeros(T)
        D_t_no_sim = np.zeros(T)
        EX_t_no_sim = np.zeros(T)

        for i in range(T):
            if i == 0:
                dstn_t_zeroth_col = np.dot(tranmat_t_zeroth_col_dense, dstn_t_zeroth_col)
            else:
                dstn_t_zeroth_col = np.dot(tranmat_ss, dstn_t_zeroth_col)

            A_t_no_sim[i] = np.dot(self.aPol_Grid.flatten(), dstn_t_zeroth_col)
            C_t_no_sim[i] = np.dot(self.cPol_Grid.flatten(), dstn_t_zeroth_col)
            D_t_no_sim[i] = np.dot(self.dPol_Grid.flatten(), dstn_t_zeroth_col)
            EX_t_no_sim[i] = np.dot(self.exPol_Grid.flatten(), dstn_t_zeroth_col)

        J_A.T[0] = (A_t_no_sim - self.A_ss) / dx
        J_C.T[0] = (C_t_no_sim - self.C_ss) / dx
        J_D.T[0] = (D_t_no_sim - self.D_ss) / dx
        J_EX.T[0] = (EX_t_no_sim - self.EX_ss) / dx

        return J_C, J_D, J_EX, J_A

class DurableConsumerSolution(MetricObject):
    """
    SAME AS ConsumerSolution BUT WITH DURABLE CONSUMPTION FUNCTION
    A class representing the solution of a single period of a consumption-saving
    problem.  The solution must include a consumption function and marginal
    value function.

    Here and elsewhere in the code, Nrm indicates that variables are normalized
    by permanent income.

    The solution has 3 steps:
    Step 1: Compute the post-decision functions wFunc and qFunc on a grid over the post-decision states n_t and a_t
    Step 2: Solve the keeper problem on a grid over the pre-decision states n_t, m_t, where the combined EGM and upper
            envelope Algorithm is applied for each n_t.
    Step 3: Solve the adjuster problem using interpolation of the keeper value function found in step 2.

    Parameters
    ----------
    cFunc : function
        The consumption function for this period, defined over durable stock and market
        resources: c = cFunc(n,m).
    dFunc: function
        The durable consumption function for this period, defined over durable stock and
        market resources: d = dFunc(n,m)
    exFunc: function
        The total expenditure function for this period, defined over durable stock and
        market resources: ex = exFunc(n,m)
    vFunc : function
        The beginning-of-period value function for this period, defined over durable stock
        and market resources: v = vFunc(n,m).
    vPFunc: function
        The beginning-of-period marginal utility function for this period, defined over durable stock
        and market resources: uP = vPFunc(n,m).
    cFuncKeep : function
        The consumption function for this periods' keeper problem, defined over durable stock and market
        resources: c = cFunc(n,m).
    dFuncKeep: function
        The durable consumption function for this periods' keeper problem, defined over durable stock and
        market resources: d = dFunc(n,m).
    exFuncKeep: function
        The total expenditure function for this periods' keeper problem, defined over durable stock and
        market resources: ex = exFunc(n,m)
    vFuncKeep : function
        The beginning-of-period value function for this periods' keeper problem, defined over durable stock
        and market resources: v = vFunc(n,m).
    vPFuncKeep: function
        The beginning-of-period marginal utility function for this periods' keeper problem, defined over durable stock
        and market resources: uP = vPFunc(n,m).
    cFuncAdj : function
        The consumption function for this periods' adjuster problem, defined over sum of durable stock and market
        resources: c = cFunc(x).
    dFuncAdj: function
        The durable consumption function for this periods' adjuster problem, defined over sum of durable stock and
        market resources: d = dFunc(x)
    exFuncAdj: function
        The total expenditure function for this periods' adjuster problem, defined over sum of durable stock and
        market resources: ex = exFunc(x)
    vFuncAdj : function
        The beginning-of-period value function for this periods' adjuster problem, defined over sum of durable stock
        and market resources: v = vFunc(x).
    vPFuncAdj: function
        The beginning-of-period marginal utility function for this periods' adjuster problem, defined over durable stock
        and market resources: uP = vPFunc(n,m).
    adjusting : function
        The adjusting function for this period indicates if for a given durable stock and market resources, the agent
        adjusts the durable stock or not: adjusting = adjusting(n,m).

    """
    distance_criteria = ["vPFunc", "vPFuncKeep", "vPFuncAdj"] #, "inv_vFunc", "inv_vFuncKeep", "inv_vFuncAdj"] # "inv_vFunc", "vFunc", "vPFunc", "adjusting"

    def __init__(
        self,
        cFunc = None,
        cFuncAdj = None,
        cFuncKeep = None,
        dFunc = None, # NEW
        dFuncAdj = None,
        dFuncKeep = None,
        exFunc = None,
        exFuncAdj = None,
        exFuncKeep = None,
        # Value Function (inverse)
        vFunc = None,
        vFuncAdj = None,
        vFuncKeep = None,
        # Marginal Utility Function
        vPFunc = None,
        vPFuncKeep = None,
        vPFuncAdj = None,
        vPFuncD = None,
        # Inverse Functions:
        inv_vFunc = None,
        inv_vFuncAdj = None,
        inv_vFuncKeep = None,
        inv_vPFunc = None,
        inv_vPFuncKeep = None,
        inv_vPFuncAdj = None,
        # Adjuster
        adjusting = None,
        mNrmGrid = None,
        nNrmGrid = None,
        mNrmMin = None,
        MPCmin = None,
        hNrm = None,
        MPCmax = None,
        aXtraGrid=None,
        # inv_vFuncKeepUnc = None,
        # cFuncKeepUnc = None,
        # exFuncKeepUnc= None,
        # # Constraint functions
        cFuncKeepCnst=None,
        dFuncKeepCnst=None,
        exFuncKeepCnst=None,
        cFuncCnst = None,
        dFuncCnst = None,
        exFuncCnst = None,
        # # Comparing upper envelope
        qFunc=None,
        qFuncD = None,
        wFunc=None,
        aNrmGridNow=None,
        # c_egm_array=None,
        # m_egm_array=None,
        # v_egm_array=None,
        BoroCnstNat = None,
        # m_thresh_max = None,
    ):

        # Change any missing function inputs to NullFunc
        self.cFunc = cFunc if cFunc is not None else NullFunc()
        self.cFuncAdj = cFuncAdj if cFuncAdj is not None else NullFunc()
        self.cFuncKeep = cFuncKeep if cFuncKeep is not None else NullFunc()
        self.dFunc = dFunc if dFunc is not None else NullFunc() # NEW
        self.dFuncAdj = dFuncAdj if dFuncAdj is not None else NullFunc()
        self.dFuncKeep = dFuncKeep if dFuncKeep is not None else NullFunc()
        self.exFunc = exFunc if exFunc is not None else NullFunc() # NEW
        self.exFuncAdj = exFuncAdj if exFuncAdj is not None else NullFunc()
        self.exFuncKeep = exFuncKeep if exFuncKeep is not None else NullFunc()
        self.vFunc = vFunc if vFunc is not None else NullFunc()
        self.vFuncAdj = vFuncAdj if vFuncAdj is not None else NullFunc()
        self.vFuncKeep = vFuncKeep if vFuncKeep is not None else NullFunc()
        self.vPFunc = vPFunc if vPFunc is not None else NullFunc()
        self.vPFuncKeep = vPFuncKeep if vPFuncKeep is not None else NullFunc()
        self.vPFuncAdj = vPFuncAdj if vPFuncAdj is not None else NullFunc()
        self.vPFuncD = vPFuncD if vPFuncD is not None else NullFunc()
        self.inv_vFunc = inv_vFunc if inv_vFunc is not None else NullFunc()
        self.inv_vFuncAdj = inv_vFuncAdj if inv_vFuncAdj is not None else NullFunc()
        self.inv_vFuncKeep = inv_vFuncKeep if inv_vFuncKeep is not None else NullFunc()
        self.inv_vPFunc = inv_vPFunc if inv_vPFunc is not None else NullFunc()
        self.inv_vPFuncKeep = inv_vPFuncKeep if inv_vPFuncKeep is not None else NullFunc()
        self.inv_vPFuncAdj = inv_vPFuncAdj if inv_vPFuncAdj is not None else NullFunc()
        self.adjusting = adjusting if adjusting is not None else NullFunc()
        self.mNrmGrid = mNrmGrid if mNrmGrid is not None else NullFunc()
        self.nNrmGrid = nNrmGrid if nNrmGrid is not None else NullFunc()
        self.mNrmMin = mNrmMin
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax
        self.aXtraGrid = aXtraGrid
        # self.inv_vFuncKeepUnc = inv_vFuncKeepUnc if inv_vFuncKeepUnc is not None else NullFunc()
        # self.cFuncKeepUnc = cFuncKeepUnc if cFuncKeepUnc is not None else NullFunc()
        # self.exFuncKeepUnc = exFuncKeepUnc if exFuncKeepUnc is not None else NullFunc()
        # # Constraint functions
        self.cFuncKeepCnst = cFuncKeepCnst if cFuncKeepCnst is not None else NullFunc()
        self.dFuncKeepCnst = dFuncKeepCnst if dFuncKeepCnst is not None else NullFunc()
        self.exFuncKeepCnst = exFuncKeepCnst if exFuncKeepCnst is not None else NullFunc()
        self.cFuncCnst = cFuncCnst if cFuncCnst is not None else NullFunc()
        self.dFuncCnst = dFuncCnst if dFuncCnst is not None else NullFunc()
        self.exFuncCnst = exFuncCnst if exFuncCnst is not None else NullFunc()
        # # Comparing upper envelope
        # self.c_egm_array = c_egm_array if c_egm_array is not None else NullFunc()
        # self.m_egm_array = m_egm_array if m_egm_array is not None else NullFunc()
        # self.v_egm_array = v_egm_array if v_egm_array is not None else NullFunc()
        self.qFunc = qFunc if qFunc is not None else NullFunc()
        self.qFuncD = qFuncD if qFuncD is not None else NullFunc()
        self.wFunc = wFunc if wFunc is not None else NullFunc()
        self.aNrmGridNow = aNrmGridNow if aNrmGridNow is not None else NullFunc()
        self.BoroCnstNat = BoroCnstNat if BoroCnstNat is not None else NullFunc()
        # self.m_thresh_max = m_thresh_max

def solve_DurableConsumer(
        solution_next,
        IncShkDstn,
        # All sorts of parameters
        alpha,  # Cobb-Douglas parameter for non-durable good consumption in utility function
        dDepr,  # Depreciation Rate of Durable Stock
        adjC,  # Adjustment costs
        d_ubar,  # Minimum durable stock for utility function
        CRRA,
        DiscFac,
        LivPrb,
        Rfree,
        PermGroFac,
        # Grids:
        mNrmGrid, #mNrmCount would be enough as well
        nNrmGrid,
        xNrmGrid, #xNrmCount would be enough as well
        grid_type,
        NestFac,
        # Borrowing Constraint:
        BoroCnstdNrm, # Cannot have negative durable Stock
        tol, # tolerance for optimization function and when to adjust vs keep
        BoroCnstArt,
        AdjX,
        expected_BOOL,
        aXtraGrid,
        UpperEnvelope,
        Limit,
        extrap_method,
):
    ####################################################################################################################
    # 1. Update utility functions:
    # i. U(C,D)
    # global cFuncAdjUnc
    u_inner = lambda C, D, d_ubar, alpha: C ** alpha * (D + d_ubar) ** (1 - alpha)
    CRRAutility = lambda C, D: utility(u_inner(C, D, d_ubar, alpha), CRRA)

    # ii. Inverse U(C, D)
    CRRAutility_inv = lambda C, D: utility_inv(CRRAutility(C, D), CRRA)

    # iii. uPC U(C,D) wrt C
    CRRAutilityP = lambda C, D: ((alpha * C ** (alpha * (1 - CRRA) - 1)) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))

    # iv. Inverse uPC U(C,D) wrt C
    # CRRAutilityP_inv = lambda C, D: utilityP_inv(CRRAutilityP(C, D), CRRA)
    CRRAutilityP_inv = lambda C, D: (
                (C / (alpha * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA)))) ** (1 / (alpha * (1 - CRRA) - 1)))

    # v. uPD U(C,D) wrt D
    CRRAutilityPD = lambda C, D: (
                (1 - alpha) * (C ** (alpha * (1 - CRRA))) * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA) - 1))

    ### Create Functions for root-finding
    def rootfinding_function(a_root, D):
        C = CRRAutilityP_inv(qFunc(D, a_root), D)
        # print(C)
        Term1 = CRRAutilityPD(C, D)
        Term2 = CRRAutilityP(C, D)  # Same as qFunc(D, a_root)
        Term3 = qFuncD(D, a_root)
        return Term1 - Term2 + Term3

    # ### Define Function:
    # def rootfinding_function_kink(D):
    #     C = CRRAutilityP_inv(qFunc(D, 0.0), D)
    #     # print(C)
    #     Term1 = CRRAutilityPD(C, D)
    #     Term2 = CRRAutilityP(C, D)  # Same as qFunc(D, a_root)
    #     Term3 = qFuncD(D, 0.0)
    #     return Term1 - Term2 + Term3

    ### Define Function:
    def rootfinding_function_kink(D):
        C = CRRAutilityP_inv(qFunc(D, mNrmMinNow), D)
        # print(C)
        Term1 = CRRAutilityPD(C, D)
        Term2 = CRRAutilityP(C, D)  # Same as qFunc(D, a_root)
        Term3 = qFuncD(D, 0.0)
        return Term1 - Term2 + Term3
    
    def max_constraint(X, d):
        return CRRAutility(X - d, d) + utility_inv(inv_wFunc(d, 0), CRRA)
    ####################################################################################################################
    # print('new period')
    # 1) Shock values:
    ShkPrbsNext = IncShkDstn.pmv
    PermShkValsNext = IncShkDstn.atoms[0]
    TranShkValsNext = IncShkDstn.atoms[1]
    iShock = len(PermShkValsNext)

    # 2. Unpack next period's solution
    inv_vFunc_next = solution_next.inv_vFunc
    vPFuncNext = solution_next.vPFunc
    cFuncNext = solution_next.cFunc
    dFuncNext = solution_next.dFunc
    vPFuncDNext = solution_next.vPFuncD

    # # Constraint functions
    cFuncKeepCnst = solution_next.cFuncKeepCnst
    dFuncKeepCnst = solution_next.dFuncKeepCnst
    exFuncKeepCnst = solution_next.exFuncKeepCnst
    cFuncCnst = solution_next.cFuncCnst
    dFuncCnst = solution_next.dFuncCnst
    exFuncCnst = solution_next.exFuncCnst

    ### NATURAL BORROWING CONSTRAINT START
    # NATURAL BORROWING CONSTRAINT. Similar to def def_BoroCnst(self, BoroCnstArt):
    PermShkMinNext = np.min(PermShkValsNext)
    TranShkMinNext = np.min(TranShkValsNext)
    DiscFacEff = DiscFac * LivPrb
    WorstIncPrb = np.sum(
        ShkPrbsNext[
            (PermShkValsNext * TranShkValsNext)
            == (PermShkMinNext * TranShkMinNext)
            ]
    )
    # Update the bounding MPCs and PDV of human wealth:
    if Limit == True:
        if alpha <1:
            gam = (alpha/(1 - alpha)) * (Rfree - 1 + dDepr)/Rfree ## Includes Durables
            Additional = (1 + gam) / (1 + gam*(Rfree - 1 + dDepr)/Rfree)
        else:
            gam = 1
            Additional = 1
    else:
        Additional = 1

    PatFac = (((Rfree * DiscFacEff) ** (1.0 / CRRA)) / Rfree) #* ((1 + gam)/(1 + gam*(Rfree - 1 + dDepr)/Rfree)) ## Includes Durables
    MPCminNow = (1.0 / (1.0 + PatFac / solution_next.MPCmin)) * Additional

    Ex_IncNext = np.dot(
        ShkPrbsNext, TranShkValsNext * PermShkValsNext
    )
    hNrmNow = (
            PermGroFac / Rfree * (Ex_IncNext + solution_next.hNrm)
    )
    MPCmaxNow = 1.0 / (
            1.0
            + (WorstIncPrb ** (1.0 / CRRA))
            * PatFac
            / solution_next.MPCmax
    )

    if Limit == True:
        if alpha < 1:
            # Limit Function (Frictionless case)
            LimitexFunc1D = lambda x: MPCminNow * (hNrmNow + x)
            LimitcFunc1D = lambda x: MPCminNow * (hNrmNow + x) * (gam/(1 + gam))
            LimitdFunc1D = lambda x: MPCminNow * (hNrmNow + x) * (1/(1 + gam))

            LimitexFunc2D = lambda n, m: MPCminNow * (hNrmNow + n *(1 - adjC) + m)
            LimitcFunc2D = lambda n, m: MPCminNow * (hNrmNow + n *(1 - adjC) + m) * (gam/(1 + gam))
            LimitdFunc2D = lambda n, m: MPCminNow * (hNrmNow + n *(1 - adjC) + m) * (1/(1 + gam))

            # Limit Gradient (Frictionless case)
            LimitexGrad1D = lambda x: [MPCminNow * np.ones_like(x)]
            LimitcGrad1D = lambda x: [MPCminNow * np.ones_like(x) * (gam/(1 + gam))]
            LimitdGrad1D = lambda x: [MPCminNow * np.ones_like(x) * (1/(1 + gam))]

            LimitexGrad2D = lambda n, m: LimitexGrad1D((1-adjC)*n + m)
            LimitcGrad2D = lambda n, m: LimitcGrad1D((1-adjC)*n + m)
            LimitdGrad2D = lambda n, m: LimitdGrad1D((1-adjC)*n + m)
        else:
            # Limit Function (Frictionless case)
            LimitexFunc1D = lambda x: MPCminNow * (hNrmNow + x)
            LimitcFunc1D = lambda x: MPCminNow * (hNrmNow + x)
            LimitdFunc1D = lambda x: MPCminNow * (hNrmNow + x) * 0

            LimitexFunc2D = lambda n, m: MPCminNow * (hNrmNow + n *(1 - adjC) + m)
            LimitcFunc2D = lambda n, m: MPCminNow * (hNrmNow + n *(1 - adjC) + m)
            LimitdFunc2D = lambda n, m: MPCminNow * (hNrmNow + n *(1 - adjC) + m) * 0

            # Limit Gradient (Frictionless case)
            LimitexGrad1D = lambda x: [MPCminNow * np.ones_like(x)]
            LimitcGrad1D = lambda x: [MPCminNow * np.ones_like(x)]
            LimitdGrad1D = lambda x: [MPCminNow * np.ones_like(x) * 0]

            LimitexGrad2D = lambda n, m: LimitexGrad1D((1-adjC)*n + m)
            LimitcGrad2D = lambda n, m: LimitcGrad1D((1-adjC)*n + m)
            LimitdGrad2D = lambda n, m: LimitdGrad1D((1-adjC)*n + m)

    # Calculate the minimum allowable value of money resources in this period
    BoroCnstNat = (
            (solution_next.mNrmMin - TranShkMinNext)
            * (PermGroFac * PermShkMinNext)
            / Rfree
    )

    ### Borowing Constraint with Durables
    # mNrmMin = solution_next.mNrmMin[0] # Borrowing constraint without any durable good
    # BoroCnstNat = np.ones(len(nNrmGrid)) * ((mNrmMin - TranShkMinNext) \
    #               * (PermGroFac * PermShkMinNext) / Rfree) + (1 - adjC) * (1 - dDepr) * nNrmGrid

    # if BoroCnstArt is None:
    #     mNrmMinNow = BoroCnstNat
    # else: #Take the maximum of both. As BoroCnstArt is a number and BoroCnstNat is an array
    #     mNrmMinNow = np.clip([BoroCnstNat, BoroCnstArt])
    #
    # To DO: aNrmGridNow is a matrix (n times a) as the Borrowingconstraint differs


    # TEST with BoroCnstNat = 0
    # BoroCnstNat = 0.0

    ### Adding durable
    # BoroCnstNat = nNrmGrid + BoroCnstNat
    # Note: need to be sure to handle BoroCnstArt==None appropriately.
    # In Py2, this would evaluate to 5.0:  np.max([None, 5.0]).
    # However in Py3, this raises a TypeError. Thus here we need to directly
    # address the situation in which BoroCnstArt == None:
    if BoroCnstArt is None:
        mNrmMinNow = BoroCnstNat
    else:
        mNrmMinNow = np.max([BoroCnstNat, BoroCnstArt])
    if BoroCnstNat < mNrmMinNow:
        MPCmaxEff = 1.0  # If actually constrained, MPC near limit is 1
    else:
        MPCmaxEff = MPCmaxNow
    ### NATURAL BORROWING CONSTRAINT END

    # End of Period Asset grid in this period. Compare to def prepare_to_calc_EndOfPrdvP(self):
    aNrmGridNow = np.asarray(aXtraGrid) + BoroCnstNat
    # print(BoroCnstNat)

    # # Adding a large point to aNrmGridNow (assures no kink at last grid point)
    # if alpha < 1:
    #     aNrmGridNow = np.append(aNrmGridNow, np.max(aXtraGrid) + 1000)

    # print(np.max(aNrmGridNow))

    nNrmGridNow = nNrmGrid
    dNrmGridNow = nNrmGrid

    ### Add a large point to aNrmGridNow and dNrmGridNow to assure no kinks at last grid point
    # if alpha < 1:
    #     aNrmGridNow = np.append(aNrmGridNow, np.max(aXtraGrid) + 1000)
        # dNrmGridNow = np.append(dNrmGridNow, np.max(dNrmGridNow) * 2)

    # dNrmGridNow = construct_grid(nNrmGrid[0],nNrmGrid[-1] * (1 - dDepr)/Rfree, len(nNrmGrid), grid_type, NestFac)
    # dNrmGridNow = construct_grid(nNrmGrid[0],nNrmGrid[-1] * 2, len(nNrmGrid), grid_type, NestFac)
    # print('new cycle')

    # Exogenous mNrmGrid
    mNrmGridNow = mNrmGrid + mNrmMinNow #BoroCnstNat
    mNrmGridNow = np.unique(np.append(mNrmGridNow, np.max(mNrmGrid)))

    # Exogenous xNrmGrid
    xMin = mNrmMinNow + (1 - adjC) * nNrmGridNow[0]
    xMax = mNrmGridNow[-1] + (1 - adjC) * nNrmGridNow[-1]
    xNrmGridNow = construct_grid(xMin, xMax, len(xNrmGrid), grid_type, NestFac)
    
    ####################################################################################################################
    # 3. Post decision function:
    '''
    Compute the post-decision functions $w_t$ and $q_t$ on a grid over the post-decision states $d_t, a_t$.
    End of period value function (EndOfPrdvFunc in IndShockConsumer): wFunc
    End-of-period marginal values (EndOfPrdvP  in IndShockConsumer): qFunc
    
    w_t(d_t, a_t) = \beta E[v_{t+1} (n_{t+1}, m_{t+1})].
    u_c(c_t,n_t) &= \alpha c_t^{\alpha(1 - \rho) - 1} n_t^{(1 - \alpha)(1 - \rho)} = q_t
    '''
    ### i. Initialize w and q
    # post_shape = (len(nNrmGridNow), len(aNrmGridNow))
    post_shape = (len(dNrmGridNow), len(aNrmGridNow))
    qFunc_array = np.zeros(post_shape)
    qFuncD_array = np.zeros(post_shape)
    wFunc_array = np.zeros(post_shape)
    # inv_wFunc_array_shape = (len(nNrmGridNow), len(aNrmGridNow)+1)
    inv_wFunc_array_shape = (len(dNrmGridNow), len(aNrmGridNow)+1)
    inv_wFunc_array = np.zeros(inv_wFunc_array_shape)
    if expected_BOOL == True:
        # for i_d in range(len(nNrmGridNow)):
        for i_d in range(len(dNrmGridNow)):
            qFunc_array[i_d] = (
                    DiscFac * LivPrb * Rfree * PermGroFac ** (-CRRA) *
                    expected(vp_next_vPFunc, IncShkDstn, args=(
                        vPFuncNext, aNrmGridNow, dNrmGridNow[i_d],
                        CRRA, dDepr, Rfree, PermGroFac)))
            
            qFuncD_array[i_d] = (
                    LivPrb * PermGroFac ** (-CRRA) * DiscFac * (1 - dDepr) *
                    expected(vp_next_vPFunc, IncShkDstn, args=(
                        vPFuncDNext, aNrmGridNow, dNrmGridNow[i_d],
                        CRRA, dDepr, Rfree, PermGroFac)))

            wFunc_array[i_d] = (
                    DiscFac * LivPrb * expected(vFunc_next, IncShkDstn, args=(
                utility_inv, inv_vFunc_next, aNrmGridNow, dNrmGridNow[i_d], CRRA, dDepr,
                Rfree, PermGroFac)))
            inv_wFunc_array[i_d,1:] = utility_inv(wFunc_array[i_d], CRRA)

    else:
        ### ii. Loop states
        # for i_d in range(len(nNrmGridNow)):
        for i_d in range(len(dNrmGridNow)):
            for ishock in range(iShock):
                # n_plus = np.ones(len(aNrmGridNow)) * ((1 - dDepr) * nNrmGridNow[i_d]) / (PermShkValsNext[ishock])
                n_plus = np.ones(len(aNrmGridNow)) * ((1 - dDepr) * dNrmGridNow[i_d]) / (PermShkValsNext[ishock])
                m_plus = Rfree / (PermGroFac * PermShkValsNext[ishock]) * aNrmGridNow + TranShkValsNext[ishock]

                # Use total solution. This already incorporates the optimal adjuster/keeper region
                inv_vFunc_plus_array = inv_vFunc_next(n_plus, m_plus)
                vPFunc_plus_array = CRRAutilityP(cFuncNext(n_plus, m_plus), dFuncNext(n_plus, m_plus))
                vPFuncD_plus_array = vPFuncDNext(n_plus, m_plus)
                vFunc_plus_array = utility_inv(inv_vFunc_plus_array, CRRA)

                wFunc_array[i_d] += ShkPrbsNext[ishock] * PermShkValsNext[ishock] ** (1.0 - CRRA) * PermGroFac ** (
                            1.0 - CRRA) * DiscFac * LivPrb * vFunc_plus_array  # weighted value function
                qFunc_array[i_d] += ShkPrbsNext[ishock] * PermShkValsNext[ishock] ** (-CRRA) * \
                                    DiscFac * LivPrb * Rfree * PermGroFac ** (-CRRA) * vPFunc_plus_array # weighted post decision function
                qFuncD_array[i_d] += ShkPrbsNext[ishock] * PermShkValsNext[ishock] ** (-CRRA) * \
                                     LivPrb * PermGroFac ** (-CRRA) * DiscFac * (1 - dDepr) * vPFuncD_plus_array

            inv_wFunc_array[i_d,1:] = utility_inv(wFunc_array[i_d], CRRA)

    # vi. transform post decision value function
    aNrm_temp = np.insert(aNrmGridNow, 0, BoroCnstNat) # Changed from BoroCnstNat
    inv_wFunc = LinearFast(inv_wFunc_array, [dNrmGridNow, aNrm_temp])
    wFunc = ValueFuncCRRA(inv_wFunc, CRRA)
    if alpha == 1 or dDepr == 1:
        qFuncD_array = np.zeros(post_shape)
    # ix. Interpolate and make functions
    qFunc = LinearFast(qFunc_array, [dNrmGridNow, aNrmGridNow])
    qFuncD = LinearFast(qFuncD_array, [dNrmGridNow, aNrmGridNow])

    # Check for Nan's
    if np.isnan(qFunc_array).any():
        print("Nan in qFunc")
    if np.isnan(inv_wFunc_array).any():
        print("NaN in wFunc")
    if np.isnan(qFuncD_array).any():
        print("Nan in qFuncD")
    # Check for kink at the last point
    if (qFunc_array[:,-1] > qFunc_array[:,-2]).any():
        print("qFunc is increasing at last point: increase aXtraGrid")

    ####################################################################################################################
    # 4. Solve Keeper Problem
    '''
    Solve the keeper problem on a grid over the pre-decision states $n_t,m_t$ where the combined EGM and 
    upper envelope is applied for each $n_t$.
    The total solution is created with a lower envelope over the constrained and unconstrained part.
    
    Inputs:
        invwFunc:   w Func from Post-decision function
        qFunc:      q Func from Post-decision function
        cFuncCnst:  Constrained function (from terminal solution)
        dFuncCnst:  Constrained function (from terminal solution)
        exFuncCnst: Constrained function (from terminal solution)
    
    Return:
        cFuncKeep
        dFuncKeep
        exFuncKeep
        vFuncKeep
        vPFuncKeep
    '''
    if UpperEnvelope == 'DCEGM':

        c_egm_array, m_egm_array, v_egm_array = EGM_njit(nNrmGridNow, aNrmGridNow, qFunc_array, wFunc_array,
                                                                alpha, d_ubar, CRRA)
        UpperEnvelope_Results = UpperEnvelope_njit(nNrmGridNow, aNrmGridNow, BoroCnstNat, qFunc, wFunc, alpha, d_ubar,
                                                  CRRA, c_egm_array, m_egm_array, v_egm_array)


    if UpperEnvelope == 'FUES':
        UpperEnvelope_Results = FUES_EGM(nNrmGridNow, aNrmGridNow, BoroCnstNat, qFunc, wFunc, alpha, d_ubar,
                                                                    CRRA)
        c_egm_array = UpperEnvelope_Results['c_egm_array']
        m_egm_array = UpperEnvelope_Results['m_egm_array']
        v_egm_array = UpperEnvelope_Results['v_egm_array']

    if UpperEnvelope == 'JEPPE':
        UpperEnvelope_Results = Upper_Envelope_Jeppe(nNrmGridNow, aNrmGridNow, mNrmGrid, qFunc, inv_wFunc_array,alpha, d_ubar,
                                                                    CRRA)

    cFuncKeepUnc_array = UpperEnvelope_Results['cFuncKeep_array']
    dFuncKeepUnc_array = UpperEnvelope_Results['dFuncKeep_array']
    mNrmGridKeeper = UpperEnvelope_Results['mNrmGridKeeper']
    inv_vPFuncKeep_array = UpperEnvelope_Results['inv_vPFuncKeep_array']
    inv_vFuncKeepUnc_array = UpperEnvelope_Results['inv_vFuncKeep_array']

    ### Make Functions
    exFuncKeepUnc_array = cFuncKeepUnc_array + dFuncKeepUnc_array
    cFuncKeepUnc = LinearFast(cFuncKeepUnc_array, [nNrmGridNow, mNrmGridKeeper])
    dFuncKeepUnc = LinearFast(dFuncKeepUnc_array, [nNrmGridNow, mNrmGridKeeper])
    exFuncKeepUnc = LinearFast(exFuncKeepUnc_array, [nNrmGridNow, mNrmGridKeeper])
    inv_vPFuncKeep = LinearFast(inv_vPFuncKeep_array, [nNrmGridNow, mNrmGridKeeper])
    inv_vFuncKeepUnc = LinearFast(inv_vFuncKeepUnc_array, [nNrmGridNow, mNrmGridKeeper])
    vFuncKeepUnc = ValueFuncCRRA(inv_vFuncKeepUnc, CRRA)

    # Use exogeneous mGrid for the adjuster problem (only if alpha < 1)
    vFuncmNrmGridNow = mNrmGridNow

    if np.min(cFuncKeepUnc_array)<0:
        print('error: cFuncKeepUnc is negative')

    # Make Consumption function similar to use_points_for_interpolation
    # Constraint part is the same as the terminal solution.
    cFuncKeep = LowerEnvelope2D(cFuncKeepUnc, cFuncKeepCnst, nan_bool=False)
    dFuncKeep = LowerEnvelope2D(dFuncKeepUnc, dFuncKeepCnst, nan_bool=False)
    exFuncKeep = LowerEnvelope2D(exFuncKeepUnc, exFuncKeepCnst, nan_bool=False)
    vPFuncKeep = MargValueFuncCRRA_dur(cFuncKeep, dFuncKeep, alpha, d_ubar, CRRA)

    # Create a vFunc similar to def make_vFunc(self, solution)
    # mNrmGridvFunc = construct_grid(BoroCnstArt, np.max(mNrmGridUnc), len(mNrmGridUnc), 'exp_mult', 3)[1:]
    keep_aux_shape = (len(nNrmGridNow), len(vFuncmNrmGridNow))
    inv_vFuncKeep_array = np.zeros(keep_aux_shape)
    for i_d in range(len(nNrmGridNow)):
        d_keep_aux = np.ones(len(vFuncmNrmGridNow)-1) * nNrmGridNow[i_d]

        # Compute expected value and marginal value on a grid of market resources
        cNrmKeepNow = cFuncKeep(d_keep_aux, vFuncmNrmGridNow[1:])

        aNrmKeepNow = vFuncmNrmGridNow[1:] - cNrmKeepNow
        vNrmKeepNow = CRRAutility(cNrmKeepNow, d_keep_aux) + utility_inv(inv_wFunc(d_keep_aux, aNrmKeepNow), CRRA)

        # Construct the beginning-of-period value function
        inv_vFuncKeep_array[i_d,1:] = utility_inv(vNrmKeepNow, CRRA)  # value transformed through inverse utility
        inv_vFuncKeep_array[i_d, 0] = 0

    inv_vFuncKeep = LinearFast(inv_vFuncKeep_array, [nNrmGridNow, vFuncmNrmGridNow])
    vFuncKeep = ValueFuncCRRA(inv_vFuncKeep, CRRA)
    ####################################################################################################################
    # 5. Solve Adjuster Problem
    '''
    Solve the adjuster problem using interpolation of the keeper value function found in step 4.
    In step 4, we found the optimal consumption given each combination of durable stock (n) and market resources.
    Now, we want to find the optimal value of (d) given cash on hand: m = x - d.
    
    Inputs:
    vFuncKeep
    
    Return:
    cFuncAdj
    dFuncAdj
    exFuncAdj
    vFuncAdj
    vPFuncAdj
    inv_vFuncAdj
    inv_vPFuncAdj
    '''
    #########################################################################################################
    if alpha == 1:
        cFuncAdj = cFuncKeep
        dFuncAdj = dFuncKeep
        exFuncAdj = exFuncKeep
        vFuncAdj = vFuncKeep
        vPFuncAdj = vPFuncKeep
        inv_vFuncAdj = inv_vFuncKeep
        inv_vPFuncAdj = inv_vPFuncKeep

    else:
        ###############################################################################################
        ### Using FUES for the unconstrained part and optimization problem over the constrained part:
        # i) Find kink point at which a is at borrowing constraint. Left to it, we need to find the constrained part differently.
        solution = root(rootfinding_function_kink, [0.0])  # For TwoNonDurables start at 0, tol = 1e-12)
        if solution.success == False:
            print('Kink point is not true')
            dNrmGridAdj = dNrmGridNow
            a_array = np.zeros(1)
            x_array = np.zeros(1)
            c_array = np.zeros(1)
            d_array = np.zeros(1)
            v_array = np.ones(1) * -np.inf

        else:
            d_kink = solution.x[0]
            # Grid for Unconstrained part:
            dNrmGridAdj = construct_grid(d_kink, np.max(dNrmGridNow), len(dNrmGridNow), grid_type,
                                         NestFac)
            # ii) Solve for constrained part:
            c_kink = CRRAutilityP_inv(qFunc(d_kink, 0.0), d_kink)
            x_kink = d_kink + c_kink
            x_array_aux = np.linspace(0.01, x_kink, 20)

            xCnst_array = [] #np.zeros(1) #[]
            cCnst_array = [] #np.zeros(1) #[] #np.zeros_like(x_array)
            dCnst_array = [] #np.zeros(1) #[] #np.zeros_like(x_array)
            # v_array = [] #np.zeros(1) #[] #np.zeros_like(x_array)
            x0 = 0.01
            for i_x in range(len(x_array_aux)):
                x_aux = x_array_aux[i_x]
                sol = sp.optimize.minimize(lambda d: -max_constraint(x_aux, d), x0, method='nelder-mead',
                                           options={'fatol': 1e-15})
                if sol.success == True:
                    xCnst_array = np.append(xCnst_array, x_aux)
                    dCnst_array = np.append(dCnst_array, sol.x[0])
                    cCnst_array = np.append(cCnst_array, x_aux - sol.x[0])
                    x0 = sol.x[0]
            # Append zero points
            cCnst_array = np.append(0, cCnst_array)
            dCnst_array = np.append(0, dCnst_array)
            xCnst_array = np.append(mNrmMinNow, xCnst_array)
            exCnst_array = cCnst_array + dCnst_array

        # iii) Find all roots
        a_min = np.min(aXtraGrid) #0.01 #np.min(aNrmGridNow) #0.01
        a_max = np.max(aXtraGrid) #np.max(aXtraGrid)
        N = 100

        a_array = []
        c_array = []
        x_array = []
        d_array = []
        v_array = []
        for i_d in range(len(dNrmGridAdj)):
            # a) Fix end-of-period durable stock
            d = dNrmGridAdj[i_d]

            # b) Solve for all roots
            a_root = find_roots_in_interval(rootfinding_function, a_min, a_max, d, N)
            d_aux = np.ones_like(a_root) * d

            # c) Evaluate endogeneous grid of beginning-of-period Wealth for each root --> endogeneous xNrmGrid
            c_0 = CRRAutilityP_inv(qFunc(d_aux, a_root), d_aux)
            x_0 = a_root + d_aux + c_0
            d_0 = d_aux
            ## d) Evaluate value for each root
            v_0 = CRRAutility(c_0, a_root) + wFunc(a_root, d_aux)

            ## Append to arrays
            a_array = np.append(a_array, a_root)
            c_array = np.append(c_array, c_0)
            x_array = np.append(x_array, x_0)
            d_array = np.append(d_array, d_0)
            v_array = np.append(v_array, v_0)

        ex_array = c_array + d_array
        # v) FUES
        if dDepr == 1 or adjC == 0:
            a_upper = a_array
            x_upper = x_array
            v_upper = v_array
            c_upper = c_array
            d_upper = d_array
            ex_upper = ex_array
        else:
            ### FUES PART STARTS HERE
            x_upper, v_upper, c_upper, a_upper, dela \
                = FUES(x_array, v_array, c_array, a_array)  # , M_bar=2, LB=10)
            ### FUES PART ENDS HERE
            d_upper = x_upper - a_upper - c_upper
            ex_upper = d_upper + c_upper

        ### vi) Adding constrained part and make functions
        if dDepr == 1:
            cFuncAdjUnc = LinearFast(c_upper, [x_upper])
            dFuncAdjUnc = LinearFast(d_upper, [x_upper])
            exFuncAdjUnc = LinearFast(ex_upper, [x_upper])
        
            cFuncAdjCnst = LinearFast(alpha * xNrmGridNow, [xNrmGridNow])
            dFuncAdjCnst = LinearFast((1 - alpha) * xNrmGridNow, [xNrmGridNow])
            exFuncAdjCnst = LinearFast(xNrmGridNow, [xNrmGridNow])

            cFuncAdj = LowerEnvelope(cFuncAdjUnc, cFuncAdjCnst, nan_bool=False)
            dFuncAdj = LowerEnvelope(dFuncAdjUnc, dFuncAdjCnst, nan_bool=False)
            exFuncAdj = LowerEnvelope(exFuncAdjUnc, exFuncAdjCnst, nan_bool=False)
        
        
            vPFuncAdj = MargValueFuncCRRA_dur(cFuncAdj, dFuncAdj, alpha, d_ubar,
                                          CRRA)
            
        else:
            xFuncAdj_array = np.append(xCnst_array, x_upper)
            cFuncAdj_array = np.append(cCnst_array, c_upper)
            dFuncAdj_array = np.append(dCnst_array, d_upper)
            exFuncAdj_array = np.append(exCnst_array, ex_upper)

            cFuncAdj = LinearFast(cFuncAdj_array, [xFuncAdj_array])
            dFuncAdj = LinearFast(dFuncAdj_array, [xFuncAdj_array])
            exFuncAdj = LinearFast(exFuncAdj_array, [xFuncAdj_array])
        
            vPFuncAdj = MargValueFuncCRRA_dur(cFuncAdj, dFuncAdj, alpha, d_ubar,
                                          CRRA)
        # vii) Create value Function over exogenous xNrmGridNow
        # Create a vFunc similar to def make_vFunc(self, solution)
        vFuncxNrmGridNow = xNrmGridNow
        inv_vFuncAdj_array = np.zeros_like(vFuncxNrmGridNow)
        cNrmAdjNow = cFuncAdj(vFuncxNrmGridNow[1:])
        dNrmAdjNow = dFuncAdj(vFuncxNrmGridNow[1:])
        aNrmAdjNow = vFuncxNrmGridNow[1:] - cNrmAdjNow - dNrmAdjNow
        vNrmAdjNow = CRRAutility(cNrmAdjNow, dNrmAdjNow) + utility_inv(inv_wFunc(dNrmAdjNow, aNrmAdjNow), CRRA)

        inv_vFuncAdj_array[1:] = utility_inv(vNrmAdjNow, CRRA)  # value transformed through inverse utility
        inv_vFuncAdj_array[0] = 0

        inv_vFuncAdj = LinearFast(inv_vFuncAdj_array, [vFuncxNrmGridNow])
        vFuncAdj = ValueFuncCRRA(inv_vFuncAdj, CRRA)
        inv_vPFuncAdj_array = CRRAutilityP_inv(cNrmAdjNow, dNrmAdjNow)
        inv_vPFuncAdj = LinearFast(inv_vPFuncAdj_array, [vFuncxNrmGridNow])

    ####################################################################################################################
    # 6. Create Consumption Function:
    '''
    Compares the value function for each combination of durable stock and market resources. Note that xNrmGrid is
    defined as (1-adjC)*nNrmGrid + mNrmGrid
    '''
    if alpha == 1: # Use keeper solution
        cFunc = cFuncKeep
        dFunc = dFuncKeep
        exFunc = exFuncKeep
        inv_vPFunc = inv_vPFuncKeep
        inv_vFunc = inv_vFuncKeep
        vFunc = vFuncKeep
        vPFunc = vPFuncKeep
        vPFuncD = vPFuncKeep #* (1 - adjC)
        solution_shape = (len(nNrmGridNow), len(mNrmGridNow))
        adjusting_array = np.ones(solution_shape)
        adjusting = LinearFast(adjusting_array, [nNrmGridNow, mNrmGridNow])
    else:
        # Create empty container
        solution_shape = (len(nNrmGridNow), len(mNrmGridNow))
        inv_vFuncAdjNM_array = np.zeros(solution_shape)
        for i_d in range(len(nNrmGridNow)):
            d_aux = np.ones(len(mNrmGridNow)) * nNrmGridNow[i_d]
            x_aux = mNrmGridNow + (1 - adjC) * d_aux
            inv_vFuncAdjNM_array[i_d] = inv_vFuncAdj(x_aux)
        if adjC > 0:
            mNrmGridTotal, lSuS_array = durable_adjusting_function_fast(nNrmGridNow, mNrmGridNow,
                                                                    inv_vFuncAdjNM_array, inv_vFuncKeep_array,
                                                                    tol)
            mNrmGridTotal = np.sort(np.append(mNrmGridTotal, np.max(mNrmGridTotal)-0.0001))
            # mNrmGridTotal = np.sort(np.append(mNrmGridTotal, [np.max(mNrmGridTotal)-0.0001, np.max(mNrmGridTotal)+0.0001]))
            # Check if nan's:
            if np.isnan(lSuS_array).any():
                print('Nans detected')
        else: #If AdjC = 0: Always adjust
            mNrmGridTotal = mNrmGridNow
            lSuS_array = np.ones(solution_shape) * mNrmGridTotal[0]

        #### FAST PART STARTS HERE
        # Create empty container
        shape = (len(nNrmGridNow), len(mNrmGridTotal))
        cFuncAdj_NM_array = np.zeros(shape)
        dFuncAdj_NM_array = np.zeros(shape)
        inv_vFuncAdj_NM_array = np.zeros(shape)
        vPFuncDAdj_NM_array = np.zeros(shape)

        cFuncKeep_NM_array = np.zeros(shape)
        dFuncKeep_NM_array = np.zeros(shape)
        inv_vFuncKeep_NM_array = np.zeros(shape)
        vPFuncDKeep_NM_array = np.zeros(shape)
        adjusting_array = np.ones(shape)

        for i_d in range(len(nNrmGridNow)):
            d_aux = np.ones(len(mNrmGridTotal)) * nNrmGridNow[i_d]
            x_aux = mNrmGridTotal + (1 - adjC) * d_aux

            cFuncAdj_NM_array[i_d] = cFuncAdj(x_aux)
            dFuncAdj_NM_array[i_d] = dFuncAdj(x_aux)
            cFuncKeep_NM_array[i_d] = cFuncKeep(d_aux, mNrmGridTotal)
            dFuncKeep_NM_array[i_d] = dFuncKeep(d_aux, mNrmGridTotal)

            inv_vFuncAdj_NM_array[i_d] = inv_vFuncAdj(x_aux)

            inv_vFuncKeep_NM_array[i_d] = inv_vFuncKeep(d_aux, mNrmGridTotal)
            vPFuncDAdj_NM_array[i_d] = (1 - adjC) * CRRAutilityP(cFuncAdj_NM_array[i_d],
                                                                 dFuncAdj_NM_array[i_d])  # Adjust
            # Create Lambda
            aNrmKeep_aux = mNrmGridTotal - cFuncKeep_NM_array[i_d]
            Lambda = (
                    LivPrb * PermGroFac ** (-CRRA) * DiscFac * (1 - dDepr) *
                    expected(vp_next_vPFunc, IncShkDstn, args=(
                        vPFuncDNext, aNrmKeep_aux, nNrmGridNow[i_d],
                        CRRA, dDepr, Rfree, PermGroFac))) 

            vPFuncDKeep_NM_array[i_d] = CRRAutilityPD(cFuncKeep_NM_array[i_d], dFuncKeep_NM_array[i_d]) + Lambda  # Keep

        cFunc_array, dFunc_array, exFunc_array, inv_vFunc_array, vPFuncD_array, adjusting_array = durable_solution_function_FUES_fast(
            nNrmGridNow, mNrmGridTotal, lSuS_array, adjusting_array,
            cFuncAdj_NM_array, dFuncAdj_NM_array,
            inv_vFuncAdj_NM_array, vPFuncDAdj_NM_array,
            cFuncKeep_NM_array, dFuncKeep_NM_array,
            inv_vFuncKeep_NM_array, vPFuncDKeep_NM_array)

        # if kink at last point (primary or secondary)
        for i_d in range(len(nNrmGridNow)):
            if cFunc_array[i_d, -1] < cFunc_array[i_d, -2]:
                print(i_d)
                print('cFunc becomes a decreasing function')
            if np.round(dFunc_array[i_d, -1],13) < np.round(dFunc_array[i_d, -2],13):
            # if dFunc_array[i_d, -1] < dFunc_array[i_d, -2]:
                print(i_d)
                print('dFunc becomes a decreasing function')
        ## FAST PART ENDS HERE
        ################################################################################################################
        # Create Policy Functions
        cFunc = LinearFast(cFunc_array, [nNrmGridNow, mNrmGridTotal])
        dFunc = LinearFast(dFunc_array, [nNrmGridNow, mNrmGridTotal])
        exFunc = LinearFast(exFunc_array, [nNrmGridNow, mNrmGridTotal])
        vPFuncD = LinearFast(vPFuncD_array, [nNrmGridNow, mNrmGridTotal])

    if Limit == True:
        # Add Limit Function:
        cFunc = DecayInterp(
            cFunc,
            limit_fun=LimitcFunc2D,
            limit_grad=LimitcGrad2D,
            extrap_method=extrap_method,
        )
        dFunc = DecayInterp(
            dFunc,
            limit_fun=LimitdFunc2D,
            limit_grad=LimitdGrad2D,
            extrap_method=extrap_method,
        )
        exFunc = DecayInterp(
            exFunc,
            limit_fun=LimitexFunc2D,
            limit_grad=LimitexGrad2D,
            extrap_method=extrap_method,
        )
    if alpha < 1:
        # Create (marginal) Value Functions and their Inverse
        inv_vPFunc_array = CRRAutilityP_inv(cFunc_array, dFunc_array)
        vPFunc_array = utilityP_inv(inv_vPFunc_array, CRRA)
        inv_vPFunc = LinearFast(inv_vPFunc_array, [nNrmGridNow, mNrmGridTotal])
        vPFunc = MargValueFuncCRRA_dur(cFunc, dFunc, alpha, d_ubar, CRRA) #LinearFast(vPFunc_array, [nNrmGridNow, mNrmGridTotal])
        adjusting = LinearFast(adjusting_array, [nNrmGridNow, mNrmGridTotal])
        inv_vFunc = LinearFast(inv_vFunc_array, [nNrmGrid, mNrmGridTotal])
        vFunc = ValueFuncCRRA(inv_vFunc, CRRA)
        vPFuncD = LinearFast(vPFuncD_array, [nNrmGridNow, mNrmGridTotal])

    ### If Depreciation Rate is one, we have an analytical solution for constraint part. We can construct the function
    ### Using the Lower Envelope over the constraint and unconstraint part
    if dDepr == 1 and alpha < 1:
        cFuncUnc = cFunc
        dFuncUnc = dFunc
        exFuncUnc = exFunc

        cFunc = LowerEnvelope2D(cFuncUnc, cFuncCnst, nan_bool=False)
        dFunc = LowerEnvelope2D(dFuncUnc, dFuncCnst, nan_bool=False)
        exFunc = LowerEnvelope2D(exFuncUnc, exFuncCnst, nan_bool=False)

        shape = (len(nNrmGridNow), len(vFuncmNrmGridNow))
        inv_vFunc_array = np.zeros(shape)
        for i_d in range(len(nNrmGridNow)):
            d_keep_aux = np.ones(len(vFuncmNrmGridNow) - 1) * nNrmGridNow[i_d]

            # Compute expected value and marginal value on a grid of market resources
            cNrmNow = cFunc(d_keep_aux, vFuncmNrmGridNow[1:])
            dNrmNow = dFunc(d_keep_aux, vFuncmNrmGridNow[1:])

            # assets depend on adjustment vs no-adjustment
            aNrmNow = np.zeros(len(cNrmNow))
            x_aux = vFuncmNrmGridNow[1:] + (1 - adjC) * d_keep_aux
            inv_vFuncKeep_aux = inv_vFuncKeep(d_keep_aux, vFuncmNrmGridNow[1:])
            inv_vFuncAdj_aux = inv_vFuncAdj(x_aux)

            for c in range(len(cNrmNow)):
                if inv_vFuncKeep_aux[c] <= inv_vFuncAdj_aux[c] + tol:
                    aNrmNow[c] = x_aux[c] - cNrmNow[c] - dNrmNow[c]
                else:
                    aNrmNow[c] = vFuncmNrmGridNow[c+1] - cNrmNow[c]

            vNrmNow = CRRAutility(cNrmNow, dNrmNow) + utility_inv(inv_wFunc(dNrmNow, aNrmNow),
                                                                             CRRA)
            # Construct the beginning-of-period value function
            inv_vFunc_array[i_d, 1:] = utility_inv(vNrmNow,
                                                       CRRA)  # value transformed through inverse utility
            inv_vFunc_array[i_d, 0] = 0

        inv_vFunc = LinearFast(inv_vFunc_array, [nNrmGridNow, vFuncmNrmGridNow])
        vFunc = ValueFuncCRRA(inv_vFunc, CRRA)
        vPFunc = MargValueFuncCRRA_dur(cFunc, dFunc, alpha, d_ubar, CRRA)

    if cFunc(0.0, 0.0) < 0.0:
        print('error: cFunc Below zero')

    # nNrmGrid = dNrmGridNow
    ####################################################################################################################
    # print(np.max(mNrmGridNow))
    # Assemble solution
    solution = DurableConsumerSolution(
        cFunc=cFunc,
        cFuncAdj=cFuncAdj,
        cFuncKeep=cFuncKeep,
        dFunc=dFunc,
        dFuncAdj=dFuncAdj,
        dFuncKeep=dFuncKeep,
        exFunc=exFunc,
        exFuncAdj=exFuncAdj,
        exFuncKeep=exFuncKeep,
        vFunc=vFunc,
        vFuncAdj=vFuncAdj,
        vFuncKeep=vFuncKeep,
        vPFunc = vPFunc,
        vPFuncAdj=vPFuncAdj,
        vPFuncKeep = vPFuncKeep,
        vPFuncD = vPFuncD,
        inv_vFunc=inv_vFunc,
        inv_vFuncAdj=inv_vFuncAdj,
        inv_vFuncKeep=inv_vFuncKeep,
        inv_vPFunc=inv_vPFunc,
        inv_vPFuncAdj=inv_vPFuncAdj,
        inv_vPFuncKeep=inv_vPFuncKeep,
        adjusting = adjusting,
        mNrmGrid = mNrmGrid,
        nNrmGrid = nNrmGrid,
        mNrmMin = mNrmMinNow,
        MPCmin = MPCminNow,
        hNrm = hNrmNow,
        MPCmax = MPCmaxEff,
        aXtraGrid = aXtraGrid,
        # inv_vFuncKeepUnc = inv_vFuncKeepUnc,
        # cFuncKeepUnc = cFuncKeepUnc,
        # exFuncKeepUnc = exFuncKeepUnc,
        # # Constraint functions
        cFuncKeepCnst=cFuncKeepCnst,
        dFuncKeepCnst=dFuncKeepCnst,
        exFuncKeepCnst=exFuncKeepCnst,
        cFuncCnst=cFuncCnst,
        dFuncCnst=dFuncCnst,
        exFuncCnst=exFuncCnst,
        # # Comparing upper envelope
        qFunc = qFunc,
        qFuncD = qFuncD,
        wFunc = wFunc,
        aNrmGridNow = aNrmGridNow,
        # c_egm_array=c_egm_array,
        # m_egm_array=m_egm_array,
        # v_egm_array=v_egm_array,
        BoroCnstNat = BoroCnstNat,
        # m_thresh_max = m_thresh_max,
    )
    return solution


class HANKIncShkDstn(DiscreteDistribution):
    """
    A one-period distribution object for the joint distribution of income
    shocks (permanent and transitory), as modeled in the Buffer Stock Theory
    paper:
        - Lognormal, discretized permanent income shocks.
        - Transitory shocks that are a mixture of:
            - A lognormal distribution in normal times.
            - An "unemployment" shock.

    Parameters
    ----------
    sigma_Perm : float
        Standard deviation of the log- permanent shock.
    sigma_Tran : float
        Standard deviation of the log- transitory shock.
    n_approx_Perm : int
        Number of points to use in the discrete approximation of the permanent shock.
    n_approx_Tran : int
        Number of points to use in the discrete approximation of the transitory shock.
    UnempPrb : float
        Probability of the "unemployment" shock.
    IncUnemp : float
        Income shock in the "unemployment" state.
    neutral_measure : Bool, optional
        Whether to use Harmenberg's permanent-income-neutral measure. The default is False.
    seed : int, optional
        Random seed. The default is 0.

    Returns
    -------
    IncShkDstn : DiscreteDistribution
        Income shock distribution.

    """

    def __init__(
            self,
            sigma_Perm,
            sigma_Tran,
            n_approx_Perm,
            n_approx_Tran,
            UnempPrb,
            IncUnemp,
            taxrate,
            TranShkMean_Func,
            labor,
            wage,

            neutral_measure=False,
            seed=0,
    ):
        perm_dstn = LognormPermIncShk(
            sigma=sigma_Perm, n_approx=n_approx_Perm, neutral_measure=neutral_measure
        )
        tran_dstn = MixtureTranIncShk_HANK(
            sigma=sigma_Tran,
            UnempPrb=UnempPrb,
            IncUnemp=IncUnemp,
            n_approx=n_approx_Tran,
            wage=wage,
            labor=labor,
            taxrate=taxrate,
            TranShkMean_Func=TranShkMean_Func
        )

        joint_dstn = combine_indep_dstns(perm_dstn, tran_dstn)

        super().__init__(pmv=joint_dstn.pmv, atoms=joint_dstn.atoms, seed=seed)


# class MixtureTranIncShk(DiscreteDistribution):
#     """
#     A one-period distribution for transitory income shocks that are a mixture
#     between a log-normal and a single-value unemployment shock.
#
#     Parameters
#     ----------
#     sigma : float
#         Standard deviation of the log-shock.
#     UnempPrb : float
#         Probability of the "unemployment" shock.
#     IncUnemp : float
#         Income shock in the "unemployment" state.
#     n_approx : int
#         Number of points to use in the discrete approximation.
#     seed : int, optional
#         Random seed. The default is 0.
#
#     Returns
#     -------
#     TranShkDstn : DiscreteDistribution
#         Transitory income shock distribution.
#
#     """
#
#     def __init__(self, sigma, UnempPrb, IncUnemp, n_approx, seed=0):
#         dstn_approx = MeanOneLogNormal(sigma).approx(
#             n_approx if sigma > 0.0 else 1, tail_N=0
#         )
#         if UnempPrb > 0.0:
#             dstn_approx = add_discrete_outcome_constant_mean(
#                 dstn_approx, p=UnempPrb, x=IncUnemp
#             )
#
#         super().__init__(pmv=dstn_approx.pmv, X=dstn_approx.X, seed=seed)

# class MixtureTranIncShk_HANK(DiscreteDistribution):
#     """
#     A one-period distribution for transitory income shocks that are a mixture
#     between a log-normal and a single-value unemployment shock.

#     Parameters
#     ----------
#     sigma : float
#         Standard deviation of the log-shock.
#     UnempPrb : float
#         Probability of the "unemployment" shock.
#     IncUnemp : float
#         Income shock in the "unemployment" state.
#     n_approx : int
#         Number of points to use in the discrete approximation.
#     seed : int, optional
#         Random seed. The default is 0.

#     Returns
#     -------
#     TranShkDstn : DiscreteDistribution
#         Transitory income shock distribution.

#     """

#     def __init__(self, sigma, UnempPrb, IncUnemp, n_approx, wage, labor, taxrate, TranShkMean_Func, seed=0):
#         dstn_approx = MeanOneLogNormal(sigma).approx(
#             n_approx if sigma > 0.0 else 1, tail_N=0
#         )

#         if UnempPrb > 0.0:
#             dstn_approx = add_discrete_outcome_constant_mean(
#                 dstn_approx, p=UnempPrb, x=IncUnemp
#             )

#         dstn_approx.atoms = dstn_approx.atoms * TranShkMean_Func(taxrate, labor, wage)

#         super().__init__(pmv=dstn_approx.pmv, atoms=dstn_approx.atoms, seed=seed)



# class MixtureTranIncShk(DiscreteDistribution):
#     """
#     A one-period distribution for transitory income shocks that are a mixture
#     between a log-normal and a single-value unemployment shock.

#     Parameters
#     ----------
#     sigma : float
#         Standard deviation of the log-shock.
#     UnempPrb : float
#         Probability of the "unemployment" shock.
#     IncUnemp : float
#         Income shock in the "unemployment" state.
#     n_approx : int
#         Number of points to use in the discrete approximation.
#     seed : int, optional
#         Random seed. The default is 0.

#     Returns
#     -------
#     TranShkDstn : DiscreteDistribution
#         Transitory income shock distribution.

#     """

#     def __init__(self, sigma, UnempPrb, IncUnemp, n_approx, seed=0):
#         dstn_approx = MeanOneLogNormal(sigma).discretize(
#             n_approx if sigma > 0.0 else 1, method="equiprobable", tail_N=0
#         )
#         if UnempPrb > 0.0:
#             dstn_approx = add_discrete_outcome_constant_mean(
#                 dstn_approx, p=UnempPrb, x=IncUnemp
#             )

#         super().__init__(pmv=dstn_approx.pmv, atoms=dstn_approx.atoms, seed=seed)

class MixtureTranIncShk_HANK(DiscreteDistribution):
    """
    A one-period distribution for transitory income shocks that are a mixture
    between a log-normal and a single-value unemployment shock.

    Parameters
    ----------
    sigma : float
        Standard deviation of the log-shock.
    UnempPrb : float
        Probability of the "unemployment" shock.
    IncUnemp : float
        Income shock in the "unemployment" state.
    n_approx : int
        Number of points to use in the discrete approximation.
    seed : int, optional
        Random seed. The default is 0.

    Returns
    -------
    TranShkDstn : DiscreteDistribution
        Transitory income shock distribution.

    """

    def __init__(self, sigma, UnempPrb, IncUnemp, n_approx, wage, labor, taxrate, TranShkMean_Func, seed=0):
        dstn_approx = MeanOneLogNormal(sigma).discretize(
            n_approx if sigma > 0.0 else 1, method="equiprobable", tail_N=0
        )
        if UnempPrb > 0.0:
            dstn_approx = add_discrete_outcome_constant_mean(
                dstn_approx, p=UnempPrb, x=IncUnemp
            )
        dstn_approx.atoms = dstn_approx.atoms * TranShkMean_Func(taxrate, labor, wage)

        super().__init__(pmv=dstn_approx.pmv, atoms=dstn_approx.atoms, seed=seed)


