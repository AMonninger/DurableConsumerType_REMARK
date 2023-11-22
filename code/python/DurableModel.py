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
TODO:
- unpack cFunc in lifecycle
- add MPC
- replace upper envelope theorem with DCEGM
"""

import numpy as np
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_idiosyncratic_shocks,
)
from HARK.interpolation import LinearInterp, BilinearInterp

# Additional:
from HARK.core import NullFunc  # Basic HARK features
from HARK.metric import MetricObject
from copy import deepcopy
from HARK.ConsumptionSaving.ConsIndShockModel import utility
from HARK.utilities import make_grid_exp_mult

# From consav
from consav import linear_interp

# check linspace
from consav.grids import nonlinspace


########################################################################################################################
### Additional Functions we need:
def func_nopar(c, d, d_ubar, alpha, rho):  # U(C,D)
    dtot = d + d_ubar
    c_total = c**alpha * dtot ** (1.0 - alpha)
    return c_total ** (1 - rho) / (1 - rho)


def create(ufunc, use_inv_w=False):
    """create upperenvelope function from the utility function ufunc

    Args:

        ufunc (callable): utility function with *args (must be decorated with @njit)

    Returns:

        upperenvelope (callable): upperenvelope called as (grid_a,m_vec,c_vec,inv_w_vec,use_inv_w,grid_m,c_ast_vec,v_ast_vec,*args)
        use_inv_w (bool,optional): assume that the post decision value-of-choice vector is a negative inverse

    """

    # @njit
    def upperenvelope(
        grid_a,
        m_vec,
        c_vec,
        inv_w_vec,
        grid_m,
        c_ast_vec,
        v_ast_vec,
        n,
        d_ubar,
        alpha,
        rho,
    ):  # *args):
        """upperenvelope function

        Args:

            grid_a (numpy.ndarray): input, end-of-period asset vector of length Na
            m_vec (numpy.ndarray): input, cash-on-hand vector from egm of length Na
            c_vec (numpy.ndarray): input, consumption vector from egm of length Na
            inv_w_vec (numpy.ndarray): input, post decision value-of-choice vector from egm of length Na
            grid_m (numpy.ndarray): input, common grid for cash-on-hand of length Nm
            c_ast_vec (numpy.ndarray): output, consumption on common grid for cash-on-hand of length Nm
            v_ast_vec (numpy.ndarray): output, value-of-choice on common grid for cash-on-hand of length Nm
            *args: additional arguments to the utility function

        """

        # for given m_vec, c_vec and w_vec (coming from grid_a)
        # find the optimal consumption choices (c_ast_vec) at the common grid (grid_m)
        # using the upper envelope + also value the implied values-of-choice (v_ast_vec)

        Na = grid_a.size
        Nm = grid_m.size

        c_ast_vec[:] = 0
        v_ast_vec[:] = -np.inf

        # constraint
        # the constraint is binding if the common m is smaller
        # than the smallest m implied by EGM step (m_vec[0])

        im = 0
        while im < Nm and grid_m[im] <= m_vec[0]:
            # a. consume all
            c_ast_vec[im] = grid_m[im]

            # b. value of choice
            u = ufunc(c_ast_vec[im], n, d_ubar, alpha, rho)
            if use_inv_w:
                v_ast_vec[im] = u + (-1.0 / inv_w_vec[0])
            else:
                v_ast_vec[im] = u + inv_w_vec[0]

            im += 1

        # upper envelope
        # apply the upper envelope algorithm

        for ia in range(Na - 1):
            # a. a inteval and w slope
            a_low = grid_a[ia]
            a_high = grid_a[ia + 1]

            inv_w_low = inv_w_vec[ia]
            inv_w_high = inv_w_vec[ia + 1]

            if a_low > a_high:
                continue

            inv_w_slope = (inv_w_high - inv_w_low) / (a_high - a_low)

            # b. m inteval and c slope
            m_low = m_vec[ia]
            m_high = m_vec[ia + 1]

            c_low = c_vec[ia]
            c_high = c_vec[ia + 1]

            c_slope = (c_high - c_low) / (m_high - m_low)

            # c. loop through common grid
            for im in range(Nm):
                # i. current m
                m = grid_m[im]

                # ii. interpolate?
                interp = (m >= m_low) and (m <= m_high)
                extrap_above = ia == Na - 2 and m > m_vec[Na - 1]

                # iii. interpolation (or extrapolation)
                if interp or extrap_above:
                    # o. implied guess
                    c_guess = c_low + c_slope * (m - m_low)
                    a_guess = m - c_guess

                    # oo. implied post-decision value function
                    inv_w = inv_w_low + inv_w_slope * (a_guess - a_low)

                    # ooo. value-of-choice
                    u = ufunc(c_guess, n, d_ubar, alpha, rho)
                    if use_inv_w:
                        v_guess = u + (-1 / inv_w)
                    else:
                        v_guess = u + inv_w

                    # oooo. update
                    if v_guess > v_ast_vec[im]:
                        v_ast_vec[im] = v_guess
                        c_ast_vec[im] = c_guess

    return upperenvelope


negm_upperenvelope = create(func_nopar, use_inv_w=True)


def obj_last_period(d, x, d_ubar, alpha, rho):
    """objective function in last period"""

    # implied consumption (rest)
    c = x - d

    return -func_nopar(c, d, d_ubar, alpha, rho)


def obj_adj(d, x, inv_v_keep, grid_d, grid_m):
    """evaluate bellman equation"""

    # a. cash-on-hand
    m = x - d

    # b. durables
    n = d

    # c. value-of-choice
    return -linear_interp.interp_2d(
        grid_d, grid_m, inv_v_keep, n, m
    )  # we are minimizing


def optimizer(
    obj, a, b, args=(), tol=1e-6
):  # making tolerance smaller doesn't change anything
    """golden section search optimizer

    Args:

        obj (callable): 1d function to optimize over
        a (double): minimum of starting bracket
        b (double): maximum of starting bracket
        args (tuple): additional arguments to the objective function
        tol (double,optional): tolerance

    Returns:

        (float): optimization result

    """

    inv_phi = (np.sqrt(5) - 1) / 2  # 1/phi
    inv_phi_sq = (3 - np.sqrt(5)) / 2  # 1/phi^2

    # a. distance
    dist = b - a
    if dist <= tol:
        return (a + b) / 2

    # b. number of iterations
    n = int(np.ceil(np.log(tol / dist) / np.log(inv_phi)))

    # c. potential new mid-points
    c = a + inv_phi_sq * dist
    d = a + inv_phi * dist
    yc = obj(c, *args)
    yd = obj(d, *args)

    # d. loop
    for _ in range(n - 1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            dist = inv_phi * dist
            c = a + inv_phi_sq * dist
            yc = obj(c, *args)
        else:
            a = c
            c = d
            yc = yd
            dist = inv_phi * dist
            d = a + inv_phi * dist
            yd = obj(d, *args)

    # e. return
    if yc < yd:
        return (a + d) / 2
    else:
        return (c + b) / 2


def construct_grid(Min, Max, Count, grid_type, NestFac):
    """
    Based on: construct_assets_grid

    Constructs grids in different grid-types.
    Options are: linear or multi-exponentially

    Parameters
    ----------
    Min:                  float
        Minimum value for the grid
    Max:                  float
        Maximum value for the grid
    Count:                 int
        Size of the grid
    # aXtraExtra:                [float]
    #     Extra values for the a-grid.
    # exp_nest:               int
    #     Level of nesting for the exponentially spaced grid

    Returns
    -------
    Grid:     np.ndarray
        Base array of values for the post-decision-state grid.
    """
    # Unpack the parameters
    Min = Min
    Max = Max
    Count = Count
    # aXtraExtra = parameters.aXtraExtra
    grid_type = grid_type
    exp_nest = NestFac

    # Set up post decision state grid:
    Grid = None
    if grid_type == "linear":
        Grid = np.linspace(Min, Max, Count)
    elif grid_type == "exp_mult":
        Grid = make_grid_exp_mult(ming=Min, maxg=Max, ng=Count, timestonest=exp_nest)
    elif grid_type == "nonlinear":
        Grid = nonlinspace(Min, Max, Count, 1.1)
    else:
        raise Exception(
            "grid_type not recognized in __init__."
            + "Please ensure grid_type is 'linear', 'nonlinear', or 'exp_mult'"
        )

    # Add in additional points for the grid:
    # for a in aXtraExtra:
    #     if a is not None:
    #         if a not in aXtraGrid:
    #             j = aXtraGrid.searchsorted(a)
    #             aXtraGrid = np.insert(aXtraGrid, j, a)

    return Grid


# For graphs:
### Plot decision function of adjusting for each n and m grid
import ipywidgets as widgets
from matplotlib import cm
import matplotlib.pyplot as plt


def decision_function(model):
    widgets.interact(
        _decision_functions,
        model=widgets.fixed(model),
        t=widgets.Dropdown(
            description="t", options=list(range(model.T_cycle + 1)), value=0
        ),
        name=widgets.Dropdown(
            description="name",
            options=["discrete", "total", "adj", "keep"],
            value="discrete",
        ),
    )


def _decision_functions(model, t, name):
    if name == "discrete":
        _discrete(model, t)
    elif name == "total":
        _total(model, t)
    elif name == "adj":
        _adj(model, t)
    elif name == "keep":
        _keep(model, t)


#     elif name == 'post_decision' and t <= model.par.T-2:
#         _w(model,t)


def _discrete(model, t):
    nNrmGrid = np.linspace(model.nNrmMin, model.nNrmMax, model.nNrmCount)
    mNrmGrid = np.linspace(model.mNrmMin, model.mNrmMax, model.mNrmCount)

    n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing="ij")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    I = model.solution[t].adjusting(n, m) > 0

    x = m[I].ravel()
    y = n[I].ravel()
    ax.scatter(x, y, s=2, label="adjust")

    x = m[~I].ravel()
    y = n[~I].ravel()
    ax.scatter(x, y, s=2, label="keep")

    ax.set_title(f"optimal discrete choice ($t = {t}$)", pad=10)

    legend = ax.legend(loc="upper center", shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor("0.90")

    # g. details
    ax.grid(True)
    ax.set_xlabel("$m_t$")
    ax.set_xlim([mNrmGrid[0], mNrmGrid[-1]])
    ax.set_ylabel("$n_t$")
    ax.set_ylim([nNrmGrid[0], nNrmGrid[-1]])

    plt.show()


# def _adj(model,t):

#     # a. grids
#     xNrmGrid = np.linspace(model.xNrmMin, model.xNrmMax, model.xNrmCount)

#     # b. figure
#     fig = plt.figure(figsize=(12,6))
#     ax_b = fig.add_subplot(1,2,1)
#     ax_v = fig.add_subplot(1,2,2)

#     # c. plot consumption
#     # Consumption Functions
#     cFuncAdj_plt = np.zeros(model.xNrmCount)
#     dFuncAdj_plt = np.zeros(model.xNrmCount)
#     exFuncAdj_plt = np.zeros(model.xNrmCount)

#     # Value Functions
#     vFuncAdj_plt = np.zeros(model.xNrmCount)

#     for i in range(model.xNrmCount):
#         cFuncAdj_plt[i] = model.solution[t].cFuncAdj(xNrmGrid[i])
#         dFuncAdj_plt[i] = model.solution[t].dFuncAdj(xNrmGrid[i])
#         exFuncAdj_plt[i] = model.solution[t].exFuncAdj(xNrmGrid[i])
#         # Value Functions
#         vFuncAdj_plt[i] = model.solution[t].vFuncAdj(xNrmGrid[i])

#     ax_b.plot(xNrmGrid,cFuncAdj_plt,label = "cFuncAdj", lw=2)
#     ax_b.plot(xNrmGrid,dFuncAdj_plt,label = "dFuncAdj", lw=2)
#     ax_b.plot(xNrmGrid,exFuncAdj_plt,label = "exFuncAdj", lw=2)

#     #ax_b.set_title("Consumption Functions", f'($t = {t}$)',pad=10)

#     # d. plot value function
#     ax_v.plot(xNrmGrid,vFuncAdj_plt,label = "vFuncAdj", lw=2)
#     #ax_v.set_title(f'neg. inverse $v^{{adj}}$ ($t = {t}$)',pad=10)

#     # e. details
#     for ax in [ax_b,ax_v]:
#         ax.grid(True)
#         ax.set_xlabel('$x_t$')
#         ax.set_xlim([xNrmGrid[0],xNrmGrid[-1]])

#     plt.legend()
#     plt.show()


def _adj(model, t):
    # grids
    nNrmGrid = np.linspace(model.nNrmMin, model.nNrmMax, model.nNrmCount)
    mNrmGrid = np.linspace(model.mNrmMin, model.mNrmMax, model.mNrmCount)

    # b. figure
    fig = plt.figure(figsize=(12, 6))
    ax_c = fig.add_subplot(2, 2, 1, projection="3d")
    ax_d = fig.add_subplot(2, 2, 2, projection="3d")
    ax_ex = fig.add_subplot(2, 2, 3, projection="3d")
    ax_v = fig.add_subplot(2, 2, 4, projection="3d")

    n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing="ij")

    # c. plot consumption
    shape = (model.nNrmCount, model.mNrmCount)
    cFuncAdj_plt = np.zeros(shape)
    dFuncAdj_plt = np.zeros(shape)
    exFuncAdj_plt = np.zeros(shape)
    vFuncAdj_plt = np.zeros(shape)

    for i_n in range(model.nNrmCount):
        for i_m in range(model.mNrmCount):
            x = (1 - model.adjC) * nNrmGrid[i_n] + mNrmGrid[i_m]
            cFuncAdj_plt[i_n, i_m] = model.solution[t].cFuncAdj(x)
            dFuncAdj_plt[i_n, i_m] = model.solution[t].dFuncAdj(x)
            exFuncAdj_plt[i_n, i_m] = model.solution[t].exFuncAdj(x)
            vFuncAdj_plt[i_n, i_m] = model.solution[t].vFuncAdj(x)

    ax_c.plot_surface(n, m, cFuncAdj_plt, cmap=cm.viridis, edgecolor="none")
    ax_c.set_title(f"$c^{{adj}}$ ($t = {t}$)", pad=10)

    ax_d.plot_surface(n, m, dFuncAdj_plt, cmap=cm.viridis, edgecolor="none")
    ax_d.set_title(f"$d^{{adj}}$ ($t = {t}$)", pad=10)

    ax_ex.plot_surface(n, m, exFuncAdj_plt, cmap=cm.viridis, edgecolor="none")
    ax_ex.set_title(f"$ex^{{adj}}$ ($t = {t}$)", pad=10)

    # d. plot value function
    ax_v.plot_surface(n, m, vFuncAdj_plt, cmap=cm.viridis, edgecolor="none")
    ax_v.set_title(f"neg. inverse $v^{{adj}}$ ($t = {t}$)", pad=10)

    # e. details
    for ax in [ax_c, ax_v]:
        ax.grid(True)
        ax.set_xlabel("$n_t$")
        ax.set_xlim([nNrmGrid[0], nNrmGrid[-1]])
        ax.set_ylabel("$m_t$")
        ax.set_ylim([mNrmGrid[0], mNrmGrid[-1]])
        ax.invert_xaxis()
    plt.legend()
    plt.show()


def _keep(model, t):
    # grids
    nNrmGrid = np.linspace(model.nNrmMin, model.nNrmMax, model.nNrmCount)
    mNrmGrid = np.linspace(model.mNrmMin, model.mNrmMax, model.mNrmCount)

    # b. figure
    fig = plt.figure(figsize=(12, 6))
    ax_c = fig.add_subplot(2, 2, 1, projection="3d")
    ax_d = fig.add_subplot(2, 2, 2, projection="3d")
    ax_ex = fig.add_subplot(2, 2, 3, projection="3d")
    ax_v = fig.add_subplot(2, 2, 4, projection="3d")

    n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing="ij")

    # c. plot consumption
    shape = (model.nNrmCount, model.mNrmCount)
    cFuncKeep_plt = np.zeros(shape)
    dFuncKeep_plt = np.zeros(shape)
    exFuncKeep_plt = np.zeros(shape)
    vFuncKeep_plt = np.zeros(shape)

    for i_n in range(model.nNrmCount):
        for i_m in range(model.mNrmCount):
            cFuncKeep_plt[i_n, i_m] = model.solution[t].cFuncKeep(
                nNrmGrid[i_n], mNrmGrid[i_m]
            )
            dFuncKeep_plt[i_n, i_m] = model.solution[t].dFuncKeep(
                nNrmGrid[i_n], mNrmGrid[i_m]
            )
            exFuncKeep_plt[i_n, i_m] = model.solution[t].exFuncKeep(
                nNrmGrid[i_n], mNrmGrid[i_m]
            )
            vFuncKeep_plt[i_n, i_m] = model.solution[t].vFuncKeep(
                nNrmGrid[i_n], mNrmGrid[i_m]
            )

    ax_c.plot_surface(n, m, cFuncKeep_plt, cmap=cm.viridis, edgecolor="none")
    ax_c.set_title(f"$c^{{keep}}$ ($t = {t}$)", pad=10)

    ax_d.plot_surface(n, m, dFuncKeep_plt, cmap=cm.viridis, edgecolor="none")
    ax_d.set_title(f"$d^{{keep}}$ ($t = {t}$)", pad=10)

    ax_ex.plot_surface(n, m, exFuncKeep_plt, cmap=cm.viridis, edgecolor="none")
    ax_ex.set_title(f"$ex^{{keep}}$ ($t = {t}$)", pad=10)

    # d. plot value function
    ax_v.plot_surface(n, m, vFuncKeep_plt, cmap=cm.viridis, edgecolor="none")
    ax_v.set_title(f"neg. inverse $v^{{keep}}$ ($t = {t}$)", pad=10)

    # e. details
    for ax in [ax_c, ax_v]:
        ax.grid(True)
        ax.set_xlabel("$n_t$")
        ax.set_xlim([nNrmGrid[0], nNrmGrid[-1]])
        ax.set_ylabel("$m_t$")
        ax.set_ylim([mNrmGrid[0], mNrmGrid[-1]])
        ax.invert_xaxis()
    plt.legend()
    plt.show()


def _total(model, t):
    # grids
    nNrmGrid = np.linspace(model.nNrmMin, model.nNrmMax, model.nNrmCount)
    mNrmGrid = np.linspace(model.mNrmMin, model.mNrmMax, model.mNrmCount)

    # b. figure
    fig = plt.figure(figsize=(12, 6))
    ax_c = fig.add_subplot(2, 2, 1, projection="3d")
    ax_d = fig.add_subplot(2, 2, 2, projection="3d")
    ax_ex = fig.add_subplot(2, 2, 3, projection="3d")
    ax_v = fig.add_subplot(2, 2, 4, projection="3d")

    n, m = np.meshgrid(nNrmGrid, mNrmGrid, indexing="ij")

    # c. plot consumption
    shape = (model.nNrmCount, model.mNrmCount)
    cFunc_plt = np.zeros(shape)
    dFunc_plt = np.zeros(shape)
    exFunc_plt = np.zeros(shape)
    vFunc_plt = np.zeros(shape)

    for i_n in range(model.nNrmCount):
        for i_m in range(model.mNrmCount):
            cFunc_plt[i_n, i_m] = model.solution[t].cFunc(nNrmGrid[i_n], mNrmGrid[i_m])
            dFunc_plt[i_n, i_m] = model.solution[t].dFunc(nNrmGrid[i_n], mNrmGrid[i_m])
            exFunc_plt[i_n, i_m] = model.solution[t].exFunc(
                nNrmGrid[i_n], mNrmGrid[i_m]
            )
            vFunc_plt[i_n, i_m] = model.solution[t].vFunc(nNrmGrid[i_n], mNrmGrid[i_m])

    ax_c.plot_surface(n, m, cFunc_plt, cmap=cm.viridis, edgecolor="none")
    ax_c.set_title(f"$c^{{total}}$ ($t = {t}$)", pad=10)

    ax_d.plot_surface(n, m, dFunc_plt, cmap=cm.viridis, edgecolor="none")
    ax_d.set_title(f"$d^{{total}}$ ($t = {t}$)", pad=10)

    ax_ex.plot_surface(n, m, exFunc_plt, cmap=cm.viridis, edgecolor="none")
    ax_ex.set_title(f"$ex^{{total}}$ ($t = {t}$)", pad=10)

    # d. plot value function
    ax_v.plot_surface(n, m, vFunc_plt, cmap=cm.viridis, edgecolor="none")
    ax_v.set_title(f"neg. inverse $v^{{total}}$ ($t = {t}$)", pad=10)

    # e. details
    for ax in [ax_c, ax_d, ax_ex, ax_v]:
        ax.grid(True)
        ax.set_xlabel("$n_t$")
        ax.set_xlim([nNrmGrid[0], nNrmGrid[-1]])
        ax.set_ylabel("$m_t$")
        ax.set_ylim([mNrmGrid[0], mNrmGrid[-1]])
        ax.invert_xaxis()
    plt.legend()
    plt.show()


########################################################################################################################
# Make a dictionary to specify an idiosyncratic income shocks consumer
init_durable = dict(
    init_idiosyncratic_shocks,
    **{
        "alpha": 0.9,  # Cobb-Douglas parameter for non-durable good consumption in utility function
        "dDepr": 0.1,  # Depreciation Rate of Durable Stock
        "adjC": 0.15,  # Adjustment costs
        "d_ubar": 1e-2,  # Minimum durable stock for utility function
        # For Grids
        "nNrmMin": 0.0,
        "nNrmMax": 10,  # 5,
        "nNrmCount": 100,
        "mNrmMin": 0.0,
        "mNrmMax": 10,
        "mNrmCount": 100,
        "xNrmMin": 0.0,
        "xNrmMax": 10,  # xMax = mNrmMax + (1 - adjC)* nNrmMax
        "xNrmCount": 100,
        "aNrmMin": 0.0,
        "aNrmMax": 11,  # xNrmMax+1.0
        "aNrmCount": 100,
        "BoroCnstdNrm": 0,  # Borrowing Constraint of durable goods.
        "tol": 1e-08,  # Tolerance for optimizer/ Acceptable difference before switching from adjuster to keeper
        "nNrmInitMean": 0,  # Initial mean of durable stock.
        "NestFac": 3,  # To construct grids differently
        "grid_type": "nonlinear",
    },
)


class DurableConsumerType(IndShockConsumerType):
    time_inv_ = IndShockConsumerType.time_inv_ + [
        "Rfree",
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
        "xNrmMin",
        "xNrmMax",
        "xNrmCount",
        "aNrmMin",
        "aNrmMax",
        "aNrmCount",
        "BoroCnstdNrm",
        "tol",
        "nNrmInitMean",
        "NestFac",
        "grid_type",
    ]

    """
    Adding the new state variable:
    nNrm: stock of durables normalized by permanent income
    """
    state_vars = IndShockConsumerType.state_vars + [
        "nNrm",
    ]

    def __init__(self, **kwds):  # verbose=1, quiet=False,
        params = init_durable.copy()
        params.update(kwds)
        # Initialize a basic consumer type
        IndShockConsumerType.__init__(self, **params)  # verbose=verbose, quiet=quiet,

        self.time_inv = deepcopy(self.time_inv_)

        self.def_utility_funcs()
        # Set the solver for the portfolio model, and update various constructed attributes
        self.solve_one_period = solve_DurableConsumer  # DurableConsumerSolver #make_one_period_oo_solver(DurableConsumerSolver)
        self.update()

    def def_utility_funcs(self):
        # i. U(C,D)
        self.u_inner = lambda C, D, d_ubar, alpha: C**alpha * (D + d_ubar) ** (
            1 - alpha
        )
        self.CRRAutility = lambda C, D: utility(
            self.u_inner(C, D, self.d_ubar, self.alpha), self.CRRA
        )

        # ii. uPC U(C,D) wrt C
        self.CRRAutilityP = lambda C, D: (
            (self.alpha * C ** (self.alpha * (1 - self.CRRA) - 1))
            * (D + self.d_ubar) ** ((1 - self.alpha) * (1 - self.CRRA))
        )

        # iii. Inverse uPC U(C,D) wrt C
        self.CRRAutilityP_inv = lambda C, D: (
            (
                C
                / (
                    self.alpha
                    * (D + self.d_ubar) ** ((1 - self.alpha) * (1 - self.CRRA))
                )
            )
            ** (1 / (self.alpha * (1 - self.CRRA) - 1))
        )

    def pre_solve(self):
        self.update_solution_terminal()

    def update(self):
        """
        We need to initialize multiple grids:
        1. Normalized durable stock grid: nNrmGrid
        2. Normalized market resource grid: mNrmGrid
        3. Normalized market resources + durable stock grid including adjustment costs: xNrmGrid
        4. Noamralized asset grid: aNrmGrid
        """

        self.updatenNrmGrid()
        self.updatemNrmGrid()
        self.updatexNrmGrid()
        self.updateaNrmGrid()

    def updatenNrmGrid(self):  # Grid of Normalized Durable Stock
        # self.nNrmGrid = np.linspace(self.nNrmMin, self.nNrmMax, self.nNrmCount)
        # self.nNrmGrid = nonlinspace(self.nNrmMin,self.nNrmMax,self.nNrmCount,1.1)
        self.nNrmGrid = construct_grid(
            self.nNrmMin, self.nNrmMax, self.nNrmCount, self.grid_type, self.NestFac
        )
        self.add_to_time_inv("nNrmGrid")

    def updatemNrmGrid(self):  # Grid of Normalized Market resouces if d\neq n
        # self.mNrmGrid = np.linspace(self.mNrmMin, self.mNrmMax, self.mNrmCount)
        # self.mNrmGrid = nonlinspace(self.mNrmMin, self.mNrmMax, self.mNrmCount,1.1)
        self.mNrmGrid = construct_grid(
            self.mNrmMin, self.mNrmMax, self.mNrmCount, self.grid_type, self.NestFac
        )
        self.add_to_time_inv("mNrmGrid")

    def updatexNrmGrid(self):  # x = m + (1 - Adjc) d
        # self.xNrmGrid = np.linspace(self.xNrmMin, self.xNrmMax, self.xNrmCount)
        # self.xNrmGrid = nonlinspace(self.xNrmMin, self.xNrmMax, self.xNrmCount,1.1)
        self.xNrmGrid = construct_grid(
            self.xNrmMin, self.xNrmMax, self.xNrmCount, self.grid_type, self.NestFac
        )
        self.add_to_time_inv("xNrmGrid")

    def updateaNrmGrid(self):  # Grid of Normalized Market resouces if d\neq n
        # self.aNrmGrid = np.linspace(self.aNrmMin, self.aNrmMax, self.aNrmCount)
        # self.aNrmGrid = nonlinspace(self.aNrmMin, self.aNrmMax, self.aNrmCount,1.1)
        self.aNrmGrid = construct_grid(
            self.aNrmMin, self.aNrmMax, self.aNrmCount, self.grid_type, self.NestFac
        )
        self.add_to_time_inv("aNrmGrid")

    # Solve last period
    def update_solution_terminal(self):
        """
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
        """

        # a) keeper problem: keep durable stock and consume everything else
        keep_shape = (len(self.nNrmGrid), len(self.mNrmGrid))
        cFuncKeep_array = np.zeros(keep_shape)
        dFuncKeep_array = np.zeros(keep_shape)

        for i_d in range(len(self.nNrmGrid)):
            for i_m in range(len(self.mNrmGrid)):
                cFuncKeep_array[i_d][i_m] = self.mNrmGrid[i_m]
                dFuncKeep_array[i_d][i_m] = self.nNrmGrid[i_d]
        exFuncKeep_array = cFuncKeep_array + dFuncKeep_array

        # Consumption Functions
        cFuncKeep_terminal = BilinearInterp(
            cFuncKeep_array, self.nNrmGrid, self.mNrmGrid
        )
        dFuncKeep_terminal = BilinearInterp(
            dFuncKeep_array, self.nNrmGrid, self.mNrmGrid
        )
        exFuncKeep_terminal = BilinearInterp(
            exFuncKeep_array, self.nNrmGrid, self.mNrmGrid
        )

        # Value Functions (negative inverse of utility function)
        # i) empty container
        inv_v_keep_array = np.zeros(keep_shape)
        inv_marg_u_keep_array = np.zeros(keep_shape)

        # ii) fill arrays
        for i_d in range(len(self.nNrmGrid)):
            for i_m in range(len(self.mNrmGrid)):
                if self.mNrmGrid[i_m] == 0:  # forced c = 0
                    cFuncKeep_array[i_d, i_m] = 0
                    inv_v_keep_array[i_d, i_m] = 0
                    inv_marg_u_keep_array[i_d, i_m] = 0
                    continue
                v_keep = self.CRRAutility(cFuncKeep_array[i_d, i_m], self.nNrmGrid[i_d])
                inv_v_keep_array[i_d, i_m] = -1.0 / v_keep
                inv_marg_u_keep_array[i_d, i_m] = 1.0 / self.CRRAutilityP(
                    cFuncKeep_array[i_d, i_m], self.nNrmGrid[i_d]
                )

        # iii) Make Functions
        vFuncKeep_terminal = BilinearInterp(
            inv_v_keep_array, self.nNrmGrid, self.mNrmGrid
        )
        uPFuncKeep_terminal = BilinearInterp(
            inv_marg_u_keep_array, self.nNrmGrid, self.mNrmGrid
        )

        # b) adjuster problem:
        # Possible Short-cut:
        # cFuncAdj_array = self.alpha * self.xNrmGrid
        # dFuncAdj_array = (1 - self.alpha) * self.xNrmGrid

        # Original:
        adj_shape = len(self.xNrmGrid)
        cFuncAdj_array = np.zeros(adj_shape)
        dFuncAdj_array = np.zeros(adj_shape)
        inv_v_adj_array = np.zeros(adj_shape)
        inv_marg_u_adj_array = np.zeros(adj_shape)

        for i_x in range(self.xNrmCount):
            # i. states
            x = self.xNrmGrid[i_x]

            if x == 0:  # forced c = d = 0
                dFuncAdj_array[i_x] = 0
                cFuncAdj_array[i_x] = 0
                inv_v_adj_array[i_x] = 0
                inv_marg_u_adj_array[i_x] = 0
                continue

            # special case if alpha = 1
            # if self.alpha == 1:
            #     dFuncAdj_array[i_x] = 0
            #     m = x - dFuncAdj_array[i_x]
            #     cFuncAdj_array[i_x] = cFuncKeep_terminal(dFuncAdj_array[i_x], m)
            #
            # else:

            # ii. optimal choices
            d_low = np.fmin(x / 2, 1e-8)
            d_high = np.fmin(x, self.nNrmMax)
            dFuncAdj_array[i_x] = optimizer(
                obj_last_period,
                d_low,
                d_high,
                args=(x, self.d_ubar, self.alpha, self.CRRA),
                tol=self.tol,
            )
            cFuncAdj_array[i_x] = x - dFuncAdj_array[i_x]

            v_adj = self.CRRAutility(
                self.xNrmGrid[i_x] - dFuncAdj_array[i_x], dFuncAdj_array[i_x]
            )
            inv_v_adj_array[i_x] = -1.0 / v_adj
            inv_marg_u_adj_array[i_x] = 1.0 / self.CRRAutilityP(
                cFuncAdj_array[i_x], dFuncAdj_array[i_x]
            )

        cFuncAdj_terminal = LinearInterp(self.xNrmGrid, cFuncAdj_array)
        dFuncAdj_terminal = LinearInterp(self.xNrmGrid, dFuncAdj_array)
        exFuncAdj_terminal = LinearInterp(self.xNrmGrid, self.xNrmGrid)
        vFuncAdj_terminal = LinearInterp(self.xNrmGrid, inv_v_adj_array)
        uPFuncAdj_terminal = LinearInterp(self.xNrmGrid, inv_marg_u_adj_array)

        """
        # value functions: negative inverse of utility function
        adj_shape = len(self.xNrmGrid)
        inv_v_adj = np.zeros(adj_shape)
        inv_marg_u_adj = np.zeros(adj_shape)
        # iii. optimal value
        for i_x in range(len(self.xNrmGrid)):
            if self.xNrmGrid[i_x] == 0:  # forced c = d = 0
                inv_v_adj[i_x] = 0
                inv_marg_u_adj[i_x] = 0
                continue
            v_adj = self.CRRAutility(self.xNrmGrid[i_x] - dFuncAdj_array[i_x], dFuncAdj_array[i_x])
            inv_v_adj[i_x] = -1.0 / v_adj
            inv_marg_u_adj[i_x] = 1.0 / self.CRRAutilityP(cFuncAdj_array[i_x], dFuncAdj_array[i_x])
        
        # Interpolate
        vFuncAdj_terminal = LinearInterp(self.xNrmGrid, inv_v_adj)
        uPFuncAdj_terminal = LinearInterp(self.xNrmGrid, inv_marg_u_adj)
        """
        # c) Create Consumption Function:
        # Using: x = (1 - tau)*d + m.
        cFunc_shape = (len(self.nNrmGrid), len(self.mNrmGrid))
        cFunc_array = np.zeros(cFunc_shape)
        dFunc_array = np.zeros(cFunc_shape)
        vFunc_array = np.zeros(cFunc_shape)
        uPFunc_array = np.zeros(cFunc_shape)
        adjusting_array = np.zeros(cFunc_shape)

        for i_m in range(len(self.mNrmGrid)):
            for i_d in range(len(self.nNrmGrid)):
                i_x = self.mNrmGrid[i_m] + (1 - self.adjC) * self.nNrmGrid[i_d]
                adjust = vFuncKeep_terminal(
                    self.nNrmGrid[i_d], self.mNrmGrid[i_m]
                ) - self.tol <= vFuncAdj_terminal(i_x)

                if adjust:
                    cFunc_array[i_d][i_m] = cFuncAdj_terminal(i_x)
                    dFunc_array[i_d][i_m] = dFuncAdj_terminal(i_x)
                    vFunc_array[i_d][i_m] = vFuncAdj_terminal(i_x)
                    uPFunc_array[i_d][i_m] = uPFuncAdj_terminal(i_x)
                    adjusting_array[i_d][i_m] = 1
                else:
                    cFunc_array[i_d][i_m] = cFuncKeep_terminal(
                        self.nNrmGrid[i_d], self.mNrmGrid[i_m]
                    )
                    dFunc_array[i_d][i_m] = dFuncKeep_terminal(
                        self.nNrmGrid[i_d], self.mNrmGrid[i_m]
                    )
                    vFunc_array[i_d][i_m] = vFuncKeep_terminal(
                        self.nNrmGrid[i_d], self.mNrmGrid[i_m]
                    )
                    uPFunc_array[i_d][i_m] = uPFuncKeep_terminal(
                        self.nNrmGrid[i_d], self.mNrmGrid[i_m]
                    )
                    adjusting_array[i_d][i_m] = 0
        exFunc_array = cFunc_array + dFunc_array

        # Interpolation
        cFunc_terminal = BilinearInterp(cFunc_array, self.nNrmGrid, self.mNrmGrid)
        dFunc_terminal = BilinearInterp(dFunc_array, self.nNrmGrid, self.mNrmGrid)
        exFunc_terminal = BilinearInterp(exFunc_array, self.nNrmGrid, self.mNrmGrid)
        vFunc_terminal = BilinearInterp(vFunc_array, self.nNrmGrid, self.mNrmGrid)
        uPFunc_terminal = BilinearInterp(uPFunc_array, self.nNrmGrid, self.mNrmGrid)
        adjusting_terminal = BilinearInterp(
            adjusting_array, self.nNrmGrid, self.mNrmGrid
        )

        # c)
        self.solution_terminal = DurableConsumerSolution(
            cFuncKeep=cFuncKeep_terminal,
            cFuncAdj=cFuncAdj_terminal,
            dFuncKeep=dFuncKeep_terminal,
            dFuncAdj=dFuncAdj_terminal,
            exFuncKeep=exFuncKeep_terminal,
            exFuncAdj=exFuncAdj_terminal,
            vFuncKeep=vFuncKeep_terminal,
            vFuncAdj=vFuncAdj_terminal,
            uPFuncKeep=uPFuncKeep_terminal,
            uPFuncAdj=uPFuncAdj_terminal,
            cFunc=cFunc_terminal,
            dFunc=dFunc_terminal,
            exFunc=exFunc_terminal,
            vFunc=vFunc_terminal,
            uPFunc=uPFunc_terminal,
            adjusting=adjusting_terminal,
        )

    ####################################################################################################################
    ### SIMULATION PART STARTS HERE:
    def initialize_sim(self):  # NOT CHANGED
        """
        Initialize the state of simulation attributes.  Simply calls the same
        method for IndShockConsumerType, then initializes the new states/shocks
        Adjust and Share.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.initialize_sim(self)

    def sim_one_period(self):  # NOT CHANGED
        """
        Simulates one period for this type.  Calls the methods get_mortality(), get_shocks() or
        read_shocks, get_states(), get_controls(), and get_poststates().  These should be defined for
        AgentType subclasses, except get_mortality (define its components sim_death and sim_birth
        instead) and read_shocks.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not hasattr(self, "solution"):
            raise Exception(
                "Model instance does not have a solution stored. To simulate, it is necessary"
                " to run the `solve()` method of the class first."
            )

        # Mortality adjusts the agent population
        self.get_mortality()  # Replace some agents with "newborns"

        # state_{t-1}
        for var in self.state_now:
            self.state_prev[var] = self.state_now[var]

            if isinstance(self.state_now[var], np.ndarray):
                self.state_now[var] = np.empty(self.AgentCount)
            else:
                # Probably an aggregate variable. It may be getting set by the Market.
                pass

        if self.read_shocks:  # If shock histories have been pre-specified, use those
            self.read_shocks_from_history()
        else:  # Otherwise, draw shocks as usual according to subclass-specific method
            self.get_shocks()
        self.get_states()  # Determine each agent's state at decision time
        self.get_controls()  # Determine each agent's choice or control variables based on states
        self.get_poststates()  # Move now state_now to state_prev

        # Advance time for all agents
        self.t_age = self.t_age + 1  # Age all consumers by one period
        self.t_cycle = self.t_cycle + 1  # Age all consumers within their cycle
        self.t_cycle[
            self.t_cycle == self.T_cycle
        ] = 0  # Resetting to zero for those who have reached the end

    # For simulation: Replace deceased with new agents
    def sim_birth(self, which_agents):
        """
        Create new agents to replace ones who have recently died; takes draws of
        initial aNrm and pLvl, as in ConsIndShockModel. Additionally, we need
        initial durable stock set to zero.

        Parameters
        ----------
        which_agents : np.array
            Boolean array of size AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        IndShockConsumerType.sim_birth(self, which_agents)
        # Add nNrm for birth of 0
        self.state_now["nNrm"][which_agents] = self.nNrmInitMean

    def get_states(self):  # Same as in core
        """
        Gets values of state variables for the current period.
        By default, calls transition function and assigns values
        to the state_now dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        new_states = self.transition()

        for i, var in enumerate(self.state_now):
            # a hack for now to deal with 'post-states'
            if i < len(new_states):
                self.state_now[var] = new_states[i]
        return None

    def transition(self):  # Added nNrmNow
        pLvlPrev = self.state_prev["pLvl"]
        aNrmPrev = self.state_prev["aNrm"]
        nNrmPrev = self.state_prev["nNrm"]  # NEW
        RfreeNow = self.get_Rfree()

        # Calculate new states: normalized market resources and permanent income level
        pLvlNow = pLvlPrev * self.shocks["PermShk"]  # Updated permanent income level
        # Updated aggregate permanent productivity level
        PlvlAggNow = self.state_prev["PlvlAgg"] * self.PermShkAggNow
        # "Effective" interest factor on normalized assets
        ReffNow = RfreeNow / self.shocks["PermShk"]
        bNrmNow = ReffNow * aNrmPrev  # Bank balances before labor income
        mNrmNow = bNrmNow + self.shocks["TranShk"]  # Market resources after income

        nNrmNow = (
            nNrmPrev * (1 - self.dDepr) / self.shocks["PermShk"]
        )  # ADDED DURABLE STOCK UPDATE
        return pLvlNow, PlvlAggNow, bNrmNow, mNrmNow, None, nNrmNow  # Added nNrmNow

    def get_controls(self):  # Added dNrmNow, exNrmNow
        """
        Calculates consumption for each consumer of this type using the consumption functions.

        :return:
        """

        cNrmNow = np.zeros(self.AgentCount) + np.nan
        # Added
        dNrmNow = np.zeros(self.AgentCount) + np.nan
        exNrmNow = np.zeros(self.AgentCount) + np.nan
        adjusting = np.zeros(self.AgentCount) + np.nan

        MPCnow = np.zeros(self.AgentCount) + np.nan  # TODO

        for t in range(self.T_cycle):
            these = t == self.t_cycle
            # CHANGE: cFunc has dimensions (nNrm, mNrm) Optimal c given the durable stock nNrm and market resources
            cNrmNow[these] = self.solution[t].cFunc(
                self.state_now["nNrm"][these], self.state_now["mNrm"][these]
            )
            dNrmNow[these] = self.solution[t].dFunc(
                self.state_now["nNrm"][these], self.state_now["mNrm"][these]
            )
            exNrmNow[these] = self.solution[t].exFunc(
                self.state_now["nNrm"][these], self.state_now["mNrm"][these]
            )
            adjusting[these] = self.solution[t].adjusting(
                self.state_now["nNrm"][these], self.state_now["mNrm"][these]
            )
            # cNrmNow[these], MPCnow[these] = self.solution[t].cFunc.eval_with_derivative(
            #     self.state_now['mNrm'][these]
            # )
        self.controls["cNrm"] = cNrmNow
        self.controls["dNrm"] = dNrmNow
        self.controls["exNrm"] = exNrmNow
        self.controls["adjusting"] = adjusting

        return None

    def get_poststates(self):  # Added nNrm and nLvl
        """
        Calculates end-of-period assets for each consumer of this type.

        Add dNrm
        :return:
        """
        # Poststates depend on decision:
        self.state_now["aNrm"] = np.zeros(self.AgentCount)
        xNrm = self.state_now["mNrm"] + (1 - self.adjC) * self.state_now["nNrm"]
        for i_Agent in range(self.AgentCount):
            if self.controls["adjusting"][i_Agent]:
                self.state_now["aNrm"][i_Agent] = (
                    xNrm[i_Agent]
                    - self.controls["cNrm"][i_Agent]
                    - self.controls["dNrm"][i_Agent]
                )
            else:
                self.state_now["aNrm"][i_Agent] = (
                    self.state_now["mNrm"][i_Agent] - self.controls["cNrm"][i_Agent]
                )

        # Useful in some cases to precalculate asset level
        self.state_now["aLvl"] = self.state_now["aNrm"] * self.state_now["pLvl"]

        # Add durable stocks normalized and in levels
        self.state_now["nNrm"] = self.controls[
            "dNrm"
        ]  # Durable consumption this period is equal to the stock
        self.state_now["nLvl"] = self.state_now["nNrm"] * self.state_now["pLvl"]
        # moves now to prev
        super().get_poststates()

        return None


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
    TODO: ADD uPFunc
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

    vPfunc : function
        The beginning-of-period marginal value function for this period,
        defined over market resources: vP = vPfunc(m).
    vPPfunc : function
        The beginning-of-period marginal marginal value function for this
        period, defined over market resources: vPP = vPPfunc(m).
    mNrmMin : float
        The minimum allowable market resources for this period; the consump-
        tion function (etc) are undefined for m < mNrmMin.
    hNrm : float
        Human wealth after receiving income this period: PDV of all future
        income, ignoring mortality.
    MPCmin : float
        Infimum of the marginal propensity to consume this period.
        MPC --> MPCmin as m --> infinity.
    MPCmax : float
        Supremum of the marginal propensity to consume this period.
        MPC --> MPCmax as m --> mNrmMin.

    """

    distance_criteria = ["vFunc"]

    def __init__(
        self,
        cFunc=None,
        cFuncAdj=None,
        cFuncKeep=None,
        dFunc=None,  # NEW
        dFuncAdj=None,
        dFuncKeep=None,
        exFunc=None,
        exFuncAdj=None,
        exFuncKeep=None,
        # Value Function (inverse)
        vFunc=None,
        vFuncAdj=None,
        vFuncKeep=None,
        # Inverse Utility Function
        uPFunc=None,
        uPFuncKeep=None,
        uPFuncAdj=None,
        vPfunc=None,
        vPPfunc=None,
        mNrmMin=None,
        hNrm=None,
        MPCmin=None,
        MPCmax=None,
        # Adjuster
        adjusting=None,
    ):
        # Change any missing function inputs to NullFunc
        self.cFunc = cFunc if cFunc is not None else NullFunc()
        self.cFuncAdj = cFuncAdj if cFuncAdj is not None else NullFunc()
        self.cFuncKeep = cFuncKeep if cFuncKeep is not None else NullFunc()
        self.dFunc = dFunc if dFunc is not None else NullFunc()  # NEW
        self.dFuncAdj = dFuncAdj if dFuncAdj is not None else NullFunc()
        self.dFuncKeep = dFuncKeep if dFuncKeep is not None else NullFunc()
        self.exFunc = exFunc if exFunc is not None else NullFunc()  # NEW
        self.exFuncAdj = exFuncAdj if exFuncAdj is not None else NullFunc()
        self.exFuncKeep = exFuncKeep if exFuncKeep is not None else NullFunc()
        self.vFunc = vFunc if vFunc is not None else NullFunc()
        self.vFuncAdj = vFuncAdj if vFuncAdj is not None else NullFunc()
        self.vFuncKeep = vFuncKeep if vFuncKeep is not None else NullFunc()
        self.vPfunc = vPfunc if vPfunc is not None else NullFunc()
        self.uPFunc = uPFunc if uPFunc is not None else NullFunc()
        self.uPFuncKeep = uPFuncKeep if uPFuncKeep is not None else NullFunc()
        self.uPFuncAdj = uPFuncAdj if uPFuncAdj is not None else NullFunc()
        # vPFunc = NullFunc() if vPfunc is None else vPfunc
        self.vPPfunc = vPPfunc if vPPfunc is not None else NullFunc()
        self.mNrmMin = mNrmMin
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax
        self.adjusting = adjusting if adjusting is not None else NullFunc()


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
    Rfree,
    # Grids:
    aNrmGrid,
    mNrmGrid,
    nNrmGrid,
    xNrmGrid,
    # Borrowing Constraint:
    BoroCnstdNrm,  # Cannot have negative durable Stock
    tol,  # tolerance for optimization function and when to adjust vs keep
):
    ####################################################################################################################
    # 1. Update utility functions:
    # i. U(C,D)
    u_inner = lambda C, D, d_ubar, alpha: C**alpha * (D + d_ubar) ** (1 - alpha)
    CRRAutility = lambda C, D: utility(u_inner(C, D, d_ubar, alpha), CRRA)

    # ii. uPC U(C,D) wrt C
    CRRAutilityP = lambda C, D: (
        (alpha * C ** (alpha * (1 - CRRA) - 1))
        * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA))
    )

    # iii. Inverse uPC U(C,D) wrt C
    CRRAutilityP_inv = lambda C, D: (
        (C / (alpha * (D + d_ubar) ** ((1 - alpha) * (1 - CRRA))))
        ** (1 / (alpha * (1 - CRRA) - 1))
    )

    ####################################################################################################################
    # 1) Shock values:
    ShkPrbsNext = IncShkDstn.pmv
    PermShkValsNext = IncShkDstn.atoms[0]
    TranShkValsNext = IncShkDstn.atoms[1]
    iShock = len(PermShkValsNext)

    # 2. Unpack next period's solution
    cFunc_next = solution_next.cFunc
    cFuncAdj_next = solution_next.cFuncAdj
    cFuncKeep_next = solution_next.cFuncKeep

    vFunc_next = solution_next.vFunc
    vFuncAdj_next = solution_next.vFuncAdj
    vFuncKeep_next = solution_next.vFuncKeep

    uPFunc_next = solution_next.uPFunc
    uPFuncAdj_next = solution_next.uPFuncAdj
    uPFuncKeep_next = solution_next.uPFuncKeep
    ####################################################################################################################
    # 3. Post decision function:
    """
    Compute the post-decision functions $w_t$ and $q_t$ on a grid over the post-decision states $d_t, a_t$.
    w_t(d_t, a_t) = \beta E[v_{t+1} (n_{t+1}, m_{t+1})].
    u_c(c_t,n_t) &= \alpha c_t^{\alpha(1 - \rho) - 1} n_t^{(1 - \alpha)(1 - \rho)} = q_t
    """

    # Create empty arrays
    post_shape = (len(nNrmGrid), len(aNrmGrid))
    invVKeepNext_array = np.zeros(post_shape)
    uPFuncKeepNext_array = np.zeros(post_shape)
    # invPUKeepNext_array = np.zeros(post_shape)
    invVAdjNext_array = np.zeros(len(xNrmGrid))
    uPFuncAdjNext_array = np.zeros(len(xNrmGrid))
    # invPUAdjNext_array = np.zeros(len(xNrmGrid))

    for i_d in range(len(nNrmGrid)):
        for i_m in range(len(mNrmGrid)):
            invVKeepNext_array[i_d, i_m] = vFuncKeep_next(nNrmGrid[i_d], mNrmGrid[i_m])
            uPFuncKeepNext_array[i_d, i_m] = uPFuncKeep_next(
                nNrmGrid[i_d], mNrmGrid[i_m]
            )

    for i_x in range(len(xNrmGrid)):
        invVAdjNext_array[i_x] = vFuncAdj_next(xNrmGrid[i_x])
        uPFuncAdjNext_array[i_x] = uPFuncAdj_next(xNrmGrid[i_x])

    ### i. Initialize w and q
    invwFunc_array = np.zeros(post_shape)
    qFunc_array = np.zeros(post_shape)

    w = np.zeros(post_shape)
    ### ii. Loop states
    # allocate temporary containers
    # m_plus = np.zeros(len(aNrmGrid))
    # x_plus = np.zeros(len(aNrmGrid))

    inv_v_keep_plus = np.zeros(len(aNrmGrid))
    inv_marg_u_keep_plus = np.zeros(len(aNrmGrid))
    inv_v_adj_plus = np.zeros(len(aNrmGrid))
    inv_marg_u_adj_plus = np.zeros(len(aNrmGrid))

    for i_d in range(len(nNrmGrid)):
        for ishock in range(iShock):
            n_plus = ((1 - dDepr) * nNrmGrid[i_d]) / (PermShkValsNext[ishock])
            m_plus = (
                (Rfree * aNrmGrid) / PermShkValsNext[ishock] + TranShkValsNext[ishock]
            )  # (Rfree * aNrmGrid + PermShkValsNext[ishock] * TranShkValsNext[ishock]) / ( PermShkValsNext[ishock])  # y_plus #R*a-grid + y_plus
            x_plus = m_plus + (1 - adjC) * n_plus

            # iii. prepare interpolators
            prep_keep = linear_interp.interp_2d_prep(nNrmGrid, n_plus, len(aNrmGrid))
            prep_adj = linear_interp.interp_1d_prep(len(aNrmGrid))

            # iv. interpolate
            linear_interp.interp_2d_only_last_vec_mon(
                prep_keep,
                nNrmGrid,
                mNrmGrid,
                invVKeepNext_array,
                n_plus,
                m_plus,
                inv_v_keep_plus,
            )

            linear_interp.interp_1d_vec_mon(
                prep_adj, xNrmGrid, invVAdjNext_array, x_plus, inv_v_adj_plus
            )

            linear_interp.interp_2d_only_last_vec_mon_rep(
                prep_keep,
                nNrmGrid,
                mNrmGrid,
                uPFuncKeepNext_array,
                n_plus,
                m_plus,
                inv_marg_u_keep_plus,
            )

            linear_interp.interp_1d_vec_mon(
                prep_adj, xNrmGrid, uPFuncAdjNext_array, x_plus, inv_marg_u_adj_plus
            )

            # v. Compare values
            for i_a in range(len(aNrmGrid)):
                keep = inv_v_keep_plus[i_a] > inv_v_adj_plus[i_a]
                if keep:
                    v_plus = -1 / inv_v_keep_plus[i_a]
                    marg_u_plus = 1 / inv_marg_u_keep_plus[i_a]
                else:
                    v_plus = -1 / inv_v_adj_plus[i_a]
                    marg_u_plus = 1 / inv_marg_u_adj_plus[i_a]

                # w[i_d, i_a] += ShkPrbsNext[ishock] * DiscFac * v_plus #weighted value function
                # qFunc_array[i_d, i_a] += ShkPrbsNext[ishock] * DiscFac * Rfree * marg_u_plus # weighted post decision function

                w[i_d, i_a] += (
                    ShkPrbsNext[ishock]
                    * PermShkValsNext[ishock] ** (1.0 - CRRA)
                    * DiscFac
                    * v_plus
                )  # weighted value function
                qFunc_array[i_d, i_a] += (
                    ShkPrbsNext[ishock]
                    * PermShkValsNext[ishock] ** (-CRRA)
                    * DiscFac
                    * Rfree
                    * marg_u_plus
                )  # weighted post decision function

    # vi. transform post decision value function
    invwFunc_array = -1 / w

    # ix. Interpolate and make functions
    invwFunc = BilinearInterp(invwFunc_array, nNrmGrid, aNrmGrid)
    qFunc = BilinearInterp(qFunc_array, nNrmGrid, aNrmGrid)

    ####################################################################################################################
    # 4. Solve Keeper Problem
    """
    Solve the keeper problem on a grid over the pre-decision states $n_t,m_t$ where the combined EGM and 
    upper envelope is applied for each $n_t$
    
    Inputs:
        invwFunc:   w Func from Post-decision function
        qFunc:      q Func from Post-decision function
    
    Return:
        cFuncKeep
        dFuncKeep
        exFuncKeep
        vFuncKeep
        PUKeep
    """

    # Empty container:
    keep_shape = (len(nNrmGrid), len(mNrmGrid))

    uPFuncKeep_array = np.zeros(keep_shape)
    dFuncKeep_array = np.zeros(keep_shape)

    cFuncKeep_array = np.zeros(post_shape)
    q_c = np.nan * np.zeros(post_shape)
    q_m = np.nan * np.zeros(post_shape)

    v_ast_vec = np.zeros(post_shape)
    for i_d in range(len(nNrmGrid)):
        d_keep = nNrmGrid[i_d]

        # use euler equation
        for i_a in range(len(aNrmGrid)):
            q_c[i_d, i_a] = CRRAutilityP_inv(
                qFunc(nNrmGrid[i_d], aNrmGrid[i_a]), d_keep
            )
            q_m[i_d, i_a] = aNrmGrid[i_a] + q_c[i_d, i_a]

        # upperenvelope
        negm_upperenvelope(
            aNrmGrid,
            q_m[i_d],
            q_c[i_d],
            invwFunc_array[i_d],
            mNrmGrid,
            cFuncKeep_array[i_d],
            v_ast_vec[i_d],
            d_keep,
            d_ubar,
            alpha,
            CRRA,
        )

        # negative inverse
        for i_m in range(len(mNrmGrid)):
            # invPUKeep_array[i_d, i_m] = 1 / marg_func_nopar(cFuncKeep_array[i_d, i_m], d_keep, d_ubar, alpha, CRRA)
            uPFuncKeep_array[i_d, i_m] = 1 / CRRAutilityP(
                cFuncKeep_array[i_d, i_m], d_keep
            )
            dFuncKeep_array[i_d, i_m] = nNrmGrid[i_d]
    vFuncKeep_array = -1 / v_ast_vec

    ### Make Functions
    exFuncKeep_array = cFuncKeep_array + dFuncKeep_array
    cFuncKeep = BilinearInterp(cFuncKeep_array, nNrmGrid, mNrmGrid)
    dFuncKeep = BilinearInterp(dFuncKeep_array, nNrmGrid, mNrmGrid)
    exFuncKeep = BilinearInterp(exFuncKeep_array, nNrmGrid, mNrmGrid)
    vFuncKeep = BilinearInterp(vFuncKeep_array, nNrmGrid, mNrmGrid)
    uPFuncKeep = BilinearInterp(uPFuncKeep_array, nNrmGrid, mNrmGrid)

    ####################################################################################################################
    # 5. Solve Adjuster Problem
    """
    Solve the adjuster problem using interpolation of the keeper value function found in step 4.
    In step 4, we found the optimal consumption given each combination of durable stock (n) and market resources.
    Now, we want to find the optimal value of (d) given cash on hand: m = x - d.
    
    Inputs:
    vFuncKeep
    
    Return:
    cFuncKeep
    dFuncKeep
    exFuncKeep
    vFuncKeep
    """
    # Create empty container
    adjust_shape = len(xNrmGrid)
    vFuncAdj_array = np.zeros(adjust_shape)
    uPFuncAdj_array = np.zeros(adjust_shape)
    # inv_v_adj_alt = np.zeros(adjust_shape)
    # inv_v_keep_array = vFuncKeep_array

    dFuncAdj_array = np.zeros(adjust_shape)
    cFuncAdj_array = np.zeros(adjust_shape)
    exFuncAdj_array = np.zeros(adjust_shape)

    # loop over x state
    for i_x in range(len(xNrmGrid)):
        # a. cash-on-hand
        x = xNrmGrid[i_x]
        if x == 0:
            dFuncAdj_array[i_x] = 0
            cFuncAdj_array[i_x] = 0
            vFuncAdj_array[i_x] = 0
            uPFuncAdj_array[i_x] = 0
            continue

        # b. optimal choice of d
        # special case if alpha = 1
        if alpha == 1:
            dFuncAdj_array[i_x] = 0
            m = x - dFuncAdj_array[i_x]
            cFuncAdj_array[i_x] = cFuncKeep(dFuncAdj_array[i_x], m)
            exFuncAdj_array[i_x] = cFuncAdj_array[i_x] + dFuncAdj_array[i_x]
        else:
            d_low = np.fmin(x / 2, 1e-8)
            # d_low = np.fmin(x / 2, 0)
            d_high = np.fmin(x, max(nNrmGrid))
            dFuncAdj_array[i_x] = optimizer(
                obj_adj,
                d_low,
                d_high,
                args=(x, vFuncKeep_array, nNrmGrid, mNrmGrid),
                tol=tol,
            )

            # Alternative:
            #            x0 = np.mean([d_high,d_low])
            #            sol_opt2 = sp.optimize.minimize(lambda d: -vFuncKeep(d, x - d), x0, method='nelder-mead',
            #                                             options={'fatol': 1e-15})
            #            dFuncAdj_array_alt = sol_opt2.x

            # c. optimal value
            m = (
                x - dFuncAdj_array[i_x]
            )  # This is correct, it is not: x - (1 - adjC) * dFuncAdj_array[i_x]
            cFuncAdj_array[i_x] = cFuncKeep(
                dFuncAdj_array[i_x], m
            )  # Evaluate cFunc at x and m
            exFuncAdj_array[i_x] = cFuncAdj_array[i_x] + dFuncAdj_array[i_x]

            # Add additional optimizer to reduce error: we know that
            # d/c = ((1 - alpha) / alpha) * (Rfree / (Rfree - 1 + dDepr))
            # ex - c = ((1 - alpha) / alpha) * (Rfree / (Rfree - 1 + dDepr))) * c
            # ex = ((1 - alpha) / alpha) * (Rfree / (Rfree - 1 + dDepr)) + 1) * c
            # c = ex/((1 - alpha / (alpha) * (Rfree / (Rfree - 1 + dDepr)) + 1)
            # ex = exFuncAdj_array[i_x]
            # cFuncAdj_array[i_x] = ex/(((1 - alpha) / alpha) * (Rfree / (Rfree - 1 + dDepr)) + 1)
            # dFuncAdj_array[i_x] = ex - cFuncAdj_array[i_x]

            # Add additional optimizer to reduce error. Given optimal total expenditure how to split it between d and c
            # ex = exFuncAdj_array[i_x]
            # x0 = ex - ex/(((1 - alpha) / alpha) * (Rfree / (Rfree - 1 + dDepr)) + 1)
            # b = [(0, ex)]
            # # sol_opt = sp.optimize.minimize(lambda d: -func_nopar(ex - d, d, d_ubar, alpha, CRRA), x0, bounds=b)
            # sol_opt = sp.optimize.minimize(lambda d: -(CRRAutility(ex - d, d) + invwFunc(d, x - ex)), x0, bounds=b)
            #
            # dFuncAdj_array[i_x] = sol_opt.x
            # cFuncAdj_array[i_x] = ex - dFuncAdj_array[i_x]

        # Added Borrowing Constraint:
        if BoroCnstdNrm > dFuncAdj_array[i_x]:
            dFuncAdj_array[i_x] = BoroCnstdNrm
            cFuncAdj_array[i_x] = exFuncAdj_array[i_x] - dFuncAdj_array[i_x]

        vFuncAdj_array[i_x] = -obj_adj(
            dFuncAdj_array[i_x], x, vFuncKeep_array, nNrmGrid, mNrmGrid
        )
        uPFuncAdj_array[i_x] = 1 / CRRAutilityP(
            cFuncAdj_array[i_x], dFuncAdj_array[i_x]
        )

    # Create Functions
    cFuncAdj = LinearInterp(xNrmGrid, cFuncAdj_array)
    dFuncAdj = LinearInterp(xNrmGrid, dFuncAdj_array)
    exFuncAdj = LinearInterp(xNrmGrid, exFuncAdj_array)
    vFuncAdj = LinearInterp(xNrmGrid, vFuncAdj_array)
    uPFuncAdj = LinearInterp(xNrmGrid, uPFuncAdj_array)
    ####################################################################################################################
    # 6. Create Consumption Function:
    """
    Compares the value function for each combination of durable stock and market resources. Note that xNrmGrid is
    defined as (1-adjC)*nNrmGrid + mNrmGrid
    """

    # Create empty container
    solution_shape = (len(nNrmGrid), len(mNrmGrid))
    cFunc_array = np.zeros(solution_shape)
    dFunc_array = np.zeros(solution_shape)
    vFunc_array = np.zeros(solution_shape)
    uPFunc_array = np.zeros(solution_shape)
    adjusting_array = np.zeros(solution_shape)

    for i_m in range(len(mNrmGrid)):
        for i_d in range(len(nNrmGrid)):
            x = mNrmGrid[i_m] + (1 - adjC) * nNrmGrid[i_d]
            adjust = vFuncKeep(nNrmGrid[i_d], mNrmGrid[i_m]) - tol <= vFuncAdj(x)
            if adjust:
                cFunc_array[i_d][i_m] = cFuncAdj(x)
                dFunc_array[i_d][i_m] = dFuncAdj(x)
                vFunc_array[i_d][i_m] = vFuncAdj(x)
                uPFunc_array[i_d][i_m] = uPFuncAdj(x)
                adjusting_array[i_d][i_m] = 1
            else:
                cFunc_array[i_d][i_m] = cFuncKeep(nNrmGrid[i_d], mNrmGrid[i_m])
                dFunc_array[i_d][i_m] = dFuncKeep(nNrmGrid[i_d], mNrmGrid[i_m])
                vFunc_array[i_d][i_m] = vFuncKeep(nNrmGrid[i_d], mNrmGrid[i_m])
                uPFunc_array[i_d][i_m] = uPFuncKeep(nNrmGrid[i_d], mNrmGrid[i_m])
                adjusting_array[i_d][i_m] = 0
    exFunc_array = cFunc_array + dFunc_array

    # Create Functions
    cFunc = BilinearInterp(cFunc_array, nNrmGrid, mNrmGrid)
    dFunc = BilinearInterp(dFunc_array, nNrmGrid, mNrmGrid)
    exFunc = BilinearInterp(exFunc_array, nNrmGrid, mNrmGrid)
    vFunc = BilinearInterp(vFunc_array, nNrmGrid, mNrmGrid)
    uPFunc = BilinearInterp(uPFunc_array, nNrmGrid, mNrmGrid)
    adjusting = BilinearInterp(adjusting_array, nNrmGrid, mNrmGrid)

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
        uPFunc=uPFunc,
        uPFuncAdj=uPFuncAdj,
        uPFuncKeep=uPFuncKeep,
        adjusting=adjusting,
    )
    return solution
