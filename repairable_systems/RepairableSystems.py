# -*- coding: utf-8 -*-
"""
 Theory Review

 Let N(t) be a Poisson process that represents the number of failures in the time interval $(0,t]$. If N(t) has
 intensity function lambda(t), then expected number of failures in the time interval (0,t] will be

 M(t)=E(N(t))=int_{0}^t lambda(u) du

 If the cost of minimal repair (MR) has expected cost CMR and the (perfect) preventive maintenance repair costs CPM,
 then for a given intensity function lambda(t) the optimal preventive maintenance repair should be done after every tau
 units of time, where tau is given by

 CPM/CMR = int_{0}^tau u * diff(lambda(u),u) du

 If n independent systems are observed, the likelihood function is

 L(u) = exp(-sum_{i=1}^{n} M(T_i) ) prod_{i,j} lambda(t_{ij}),

 where t_{ij} is the j-th failure time for the i-th system and T_i represents (i) the truncate time of time truncated
 systems or (ii) the time of the last failure when the system is failure truncated.

 References:
 [1] https://mpmath.googlecode.com/svn/trunk/doc/build/functions/hypergeometric.html#meijerg
 [2] https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29
 [3] Gilardoni, Gustavo L., and Enrico A. Colosimo. "Optimal maintenance time for repairable systems." Journal of
     Quality Technology 39.1 (2007): 48-53.
 [4] Pulcini, Gianpaolo. "A bounded intensity process for the reliability of repairable equipment." Journal of Quality
     Technology 33.4 (2001): 480-492.
"""

from mpmath import mp
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.integrate as integrate
from numbers import Number
import time
import sys


def newton_raphson(f, x, tol=1.0e-9, verbose=False):
    """

    :type verbose: boolean
    """
    if verbose:
        print 'Init -------'
        print 'x    = ', x
        print 'F(x) = ', f(x)

    def jacobian(f, x):
        h = 1E-04
        n = x.size
        jac = np.zeros((n, n), dtype=np.float)
        f0 = f(x)
        for i in range(n):
            temp = x[i]
            x[i] = temp + h
            f1 = f(x)
            x[i] = temp
            jac[:, i] = (f1 - f0) / h
        return jac, f0

    maxit = 30

    for i in range(maxit):
        jac, f0 = jacobian(f, x)
        if np.linalg.norm(f0) / x.size < tol: break
        dx = np.linalg.solve(jac, -f0)
        x = x + dx
        if np.linalg.norm(dx) < tol * max(max(np.abs(x)), 1): break

    if i < maxit:
        print 'Final -----'
        print 'x     = ', x
        print 'F(x)  = ', f(x)
        print 'niter = ', i + 1
        return x, i + 1

    raise Exception('Too many iterations')


def bootci_get_min_ci(x):
    x = np.sort(x)
    n = len(x)
    if n is 0: return [0, 0]
    m = int(0.95 * n) - 1
    ci = [x[0], x[m]]
    for i in range(0, n - m):
        # shift confidence interval
        if (ci[1] - ci[0]) > (x[i + m] - x[i]):
            ci = [x[i], x[i + m]]
    return ci


def bootci(nboot, bootfun, data):
    """ Random sampling with replacement (See [2]) """
    nsystems = data.numberOfSystems
    n = len(bootfun(data))  # number of statistics

    # elapsed time function
    tstart = time.time()
    def eta(): return time.time() - tstart

    ci = np.zeros((n, 2), dtype=float)
    bootstat = np.zeros(n, dtype=float)
    if nboot < 1:
        print('Bootstrapping #sampling = %d [100%%] %3.1fs (elapsed time)\r' % (nboot, eta()))
        return ci, bootstat

    # bootstrap kernel
    bootmat = np.zeros((nboot, n))
    for k in range(nboot):
        sys.stdout.flush()
        print('Bootstrapping #sampling = %d [%3d%%] %3.1fs (elapsed time)\r' % (nboot, (1E2*k/nboot), eta())),

        index = np.random.randint(nsystems, size=nsystems)
        sample = data.sample(index)
        # eval bootfun
        bootmat[k, :] = bootfun(sample)
    print('Bootstrapping #sampling = %d [100%%] %3.1fs (elapsed time)\r' % (nboot, eta()))

    bootstat = [np.mean(bootmat[:, i]) for i in range(n)]

    # find the minimal confidence interval (ci) for each statistic

    for i in range(n):
        ci[i, :] = bootci_get_min_ci(bootmat[:, i])

    return ci, bootstat


def AIC(lnL, k): return -2.0 * lnL + 2.0 * k
def AICc(lnL, k, n): return - 2.0 * lnL + 2.0 * (k + k * (k + 1.0)/(n - k - 1.0))
def BIC(lnL, k, n): return -2.0 * lnL + np.log(n) / k
def MSE(model, data):
    return 0

def lnlike(model, data):
    assert isinstance(data, RepairableData)
    si = 0
    for ti in data.censorTimes:
        si += model.ExpectedNumberOfFailures(ti)
    sij = 0
    for tij in data.allFailures:
        sij += model.intensity(tij, None, None)
    return sij - si


class MCNF(object):
    """ Mean Cumulative Number of Failures (mcnf) as a function of t (time) """
    def __init__(self, data):
        censor_times = data.censorTimes

        # get unique failure times and time interval
        t = [0]
        for i in range(data.numberOfSystems):
            # add censor times
            if censor_times[i] not in t:
                t.append(censor_times[i])
            for j in range(len(data.failures[i])):
                tij = data.failures[i][j]
                # add failure times
                if tij not in t: t.append(tij)
        t = np.sort(t)

        # mcnf[j]: mean number of cumulative failures until time t(j)
        # q[j]   : number of uncensored systems until time t(j)
        mcnf = np.zeros(len(t))
        q = np.zeros(len(t))
        for j in range(len(t)):
            q[j] = np.sum(censor_times > t[j])
            for i in range(data.numberOfSystems):
                if censor_times[i] > t[j]:
                    mcnf[j] = mcnf[j] + sum(data.failures[i] <= t[j])

        # adjust the average
        for j in range(len(q)):
            mcnf[j] = mcnf[j] / q[j] if q[j] > 0 else 0

        self.failures = t
        self.meanCumulativeNumberOfFailures = mcnf
        self.numberOfUncensoredSystems = q
        self.maxCensorTimes = np.max(censor_times)

    def plot(self, axis=plt):
        t = self.failures
        mcnf = self.meanCumulativeNumberOfFailures
        x = np.zeros(4 * (len(t) - 1))
        y = np.zeros(len(x))
        j = 0
        for i in range(len(t) - 1):
            x[j] = t[i]
            x[(j + 1):(j + 4)] = t[i + 1]
            y[j:(j + 3)] = mcnf[i]
            y[j + 3] = mcnf[i + 1]
            j += 4
        axis.plot(x, y, 'k-')
        axis.set_xlim((0, self.maxCensorTimes * 1.05))
        axis.set_ylim((-0.02, np.max(y) * 1.05))
        axis.set_title('Mean Cumulative Number of Failures')
        axis.set_xlabel('Time (x1000 hours)')
        axis.set_ylabel('Number of Failures')

    def eval(self, t):
        # bracket
        index = 0
        for i in range(len(self.failures)):
            if self.failures[i] > t:
                index = i - 1
                break
        return self.meanCumulativeNumberOfFailures[index]


class RepairableData(object):
    """ Repairable Data Model """
    # failures    : list of lists with the time of each failure for each system
    # allFailures : list of each failure
    def __init__(self, filename=None, plot=False):
        # create an empty data
        if filename is None:
            self.CMR = 0
            self.CPM = 0
            self.numberOfSystems = 0
            self.failures = []
            self.censorTimes = []
            return

        # read data from file
        with open(filename) as fid:
            values = np.fromstring(fid.readline(), dtype=np.float, sep=' ')
            self.CPM = values[0]
            self.CMR = values[1]
            rawdata = fid.readlines()
            # convert from time from hours to thousand of hours
            times = [np.fromstring(data, dtype=np.float, sep=' ') / 1000 for data in rawdata]
            self.numberOfSystems = len(times)
            # read failures
            self.failures = [t[0:-1] for t in times]       # time of failures for each system
            self.allFailures = []
            for fi in self.failures:
                for fij in fi:
                    self.allFailures.append(fij)
            self.allFailures = np.array(self.allFailures)  # array with all failure times
            self.numberOfFailures = np.sum([len(failures) for failures in self.failures])
            self.censorTimes = np.array([t[-1] for t in times])
            self.__mcnf = MCNF(self)
            if plot: self.plot()

    def mcnf(self, t):
        return self.__mcnf.plot(t)

    def plot_mcnf(self, axis=plt):
        self.__mcnf.plot(axis)

    def plot_failures(self, axis=plt):
        for i in range(self.numberOfSystems):
            axis.plot([0, self.censorTimes[i]], [i + 1, i + 1], 'b-')
            axis.plot(self.censorTimes[i], i + 1, 'yo')
            axis.plot(self.failures[i], [i + 1 for j in range(len(self.failures[i]))], 'ro')

        axis.set_ylabel('System ID')
        axis.set_xlabel('Failure Times (x1000 hours)')
        axis.set_ylim(0, self.numberOfSystems + 1)
        axis.set_title('Repairable Data (CMR: {}, CPM: {})'.format(self.CMR, self.CPM))

    def sample(self, index, plot=False):
        data = RepairableData()
        data.CMR = self.CMR
        data.CPM = self.CPM
        data.censorTimes = [self.censorTimes[i] for i in index]
        data.failures = [self.failures[i] for i in index]
        data.allFailures = []
        for fi in data.failures:
            for fij in fi:
                data.allFailures.append(fij)
        data.allFailures = np.array(self.allFailures)  # array with all failure times
        data.numberOfSystems = len(index)
        if plot:
            data.plot()
        return data


class RepairableModelPLP(object):
    """ Repairable Model PLP
        Power Law Process : lambda(t)=beta t^(beta-1)/theta^beta
    """
    def __init__(self, data):
        self.data = data

        print('# Model: PLP --------------------------------------')

        # init parameters
        (beta, theta, tau, H, ci) = self.bootstrap(data)

        # set model parameters
        self.beta = beta; self.theta = theta; self.tau = tau; self.H = H; self.ci = ci

        print 'beta  ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(beta, ci['beta'][0], ci['beta'][1])
        print 'theta ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(theta, ci['theta'][0], ci['theta'][1])
        print 'tau .............. {:9.3g} [{:9.3g}, {:9.3g}] x 1000 hours'.format(tau, ci['tau'][0], ci['tau'][1])
        print 'H ................ {:9.3g} [{:9.3g}, {:9.3g}] / 1000 hours'.format(H, ci['H'][0], ci['H'][1])

        logl = lnlike(self, data)
        print 'AIC .............. {:9.3g}'.format(AIC(logl, 2))
        print 'AICc ............. {:9.3g}'.format(AICc(logl, 2, data.numberOfSystems))
        print 'BIC .............. {:9.3g}'.format(BIC(logl, 2, data.numberOfSystems))
        print 'MSE .............. {:9.3g}'.format(MSE(self, data))


    def ExpectedNumberOfFailures(self, t, beta=None, theta=None,):
        """ Calculates N(t), the expected number of failures until time t.
            beta  :: plp parameter (optional)
            theta :: plp parameter (optional)
            Model evaluation - Expected number of failures until time t.
        """
        if beta is None: beta = self.beta
        if theta is None: theta = self.theta

        Nt = (t / theta) ** beta
        return Nt

    def ExpectedCostPerUnitOfTime(self, beta, theta, tau, data):
        """ Calculates H(t), the expected cost per unit of time
            See [Gilardoni2007] eq.(2) pp. 49
        """
        H = (data.CPM + data.CMR * self.ExpectedNumberOfFailures(tau, beta, theta)) / tau
        return H

    def intensity(self, t, beta=None, theta=None):
        # Evaluates the intensity function (lambda).
        # t     :: time
        # beta  :: PLP parameter
        # theta :: PLP parameter
        if beta is None: beta = self.beta
        if theta is None: theta = self.theta
        y = (beta / theta) * (t / theta) ** (beta - 1)
        return y

    def calc_tau(self, beta, theta, data):
        # See [Gilardoni2007] eq. (5) pp. 50
        tau = theta * (data.CPM / ((beta - 1) * data.CMR)) ** (1 / beta)
        return tau

    def gap_tau(self, beta, theta, tau, data):
        # Check the error (gap) in the current tau value.
        # See [Gilardoni2007] eq. (4) pp. 49
        gap = tau * self.intensity(tau, beta, theta) - self.ExpectedNumberOfFailures(tau, beta, theta) - data.CPM / data.CMR
        return gap

    def plot(self, axis=plt):
        tmax = np.max(self.data.censorTimes)
        t = np.linspace(0, tmax)
        Nt = self.ExpectedNumberOfFailures(t)
        axis.plot(t, Nt, label='PLP', marker='d')

    # BOOTSTRAP ====================================================================
    def bootstrap(self, data, verbose=False):
        # set parameters (full data)
        (beta, theta, tau, H) = self.bootfun(data)

        # calc confidence interval (ci) using bootstrap
        nboot = 0
        [ci, bootstat] = bootci(nboot, self.bootfun, data)

        # set confidence intervals
        ci = {'beta': ci[0], 'theta': ci[1], 'tau': ci[2], 'H': ci[3]}

        return beta, theta, tau, H, ci

    def bootfun(self, data):
        # set beta and theta using CMLE
        (beta, theta) = self.CMLE(data)

        # set tau = tau(beta,theta)
        tau = self.calc_tau(beta, theta, data)

        # set H(tau) (See eq.(2) [Gilardoni2007] pp. 49)
        H = self.ExpectedCostPerUnitOfTime(beta, theta, tau, data)

        # set output
        return beta, theta, tau, H

    # CMLE :: Conditional Maximum Likelihood Estimator =============================
    def CMLE(self, data):
        # See [Ringdon2000] pp. 210 and [Crow1975]
        M = self.CMLE_M(data)
        beta = self.CMLE_beta(M, data)
        theta = self.CMLE_theta(beta, M, data)
        return (beta, theta)

    def CMLE_M(self, data):
        # See [Ringdon2000] pp. 210
        m = np.zeros(data.numberOfSystems)
        for i in range(data.numberOfSystems):
            ti = data.failures[i]
            Ti = data.censorTimes[i]
            if len(ti) > 0:
                m[i] = len(ti) - (ti[-1] == Ti)
            else:
                m[i] = 0
        M = sum(m)
        return M

    def CMLE_beta(self, M, data):
        # See [Ringdon2000] pp. 210
        k = 0
        for i in range(data.numberOfSystems):
            ti = data.failures[i]
            Ti = data.censorTimes[i]
            k = k + np.sum(np.log(Ti / ti))
        beta = M / k
        return beta

    def CMLE_theta(self, beta, M, data):
        # See [Ringdon2000] pp. 210
        T = data.censorTimes
        theta = sum(T ** beta / M) ** (1 / beta)
        return theta


class RepairableModelGA:
    """ Repairable Model GA """
    def __init__(self, data):
        print('# Model: GA     -----------------------------------')
        self.data = data

        (self.beta, self.gamma, self.theta, self.tau, self.H, self.ci) = self.__bootstrap__(data, 'fsolve', verbose=False)

        print('beta  ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(self.beta, self.ci['beta'][0], self.ci['beta'][1]))
        print('theta ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(self.theta, self.ci['theta'][0], self.ci['theta'][1]))
        print('gamma ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(self.gamma, self.ci['gamma'][0], self.ci['gamma'][1]))
        print('tau .............. {:9.3g} [{:9.3g}, {:9.3g}] x 1000 hours'.format(self.tau, self.ci['tau'][0], self.ci['tau'][1]))
        print('H ................ {:9.3g} [{:9.3g}, {:9.3g}] / 1000 hours'.format(self.H, self.ci['H'][0], self.ci['H'][1]))

        logl = lnlike(self, data)
        print 'AIC .............. {:9.3g}'.format(AIC(logl, 3))
        print 'AICc ............. {:9.3g}'.format(AICc(logl, 3, data.numberOfSystems))
        print 'BIC .............. {:9.3g}'.format(BIC(logl, 3, data.numberOfSystems))
        print 'MSE .............. {:9.3g}'.format(MSE(self, data))

    def intensity(self, t, beta=None, gamma=None, theta=None):
        if beta is None: beta = self.beta
        if theta is None: theta = self.theta
        if gamma is None: gamma = self.gamma
        return beta * (1 - np.exp(-t ** gamma / theta))

    def __calc_tau__(self, beta, gamma, theta, data, verbose=False):
        # using a nonlinear solver (fsolve)
        # ToDo(1): Add derivative of gap (fprime)
        # ToDo(2): Check if it is better to use args instead of lambda functions
        # ToDo(3): Check if it is better to use equation (4) of Gilardoni2007 instead of special function G

        kappa = data.CPM / data.CMR
        G1223 = lambda z: mp.meijerg([[0, 1 - 1 / gamma], []], [[0], [1, -1 / gamma]], (z ** gamma) / theta)
        gap = lambda tau: kappa + beta * tau[0] * G1223(tau[0])
        x0 = [5.84]  # from ModelPLP
        tau = opt.fsolve(gap, x0, args=(), fprime=None)

        if verbose:
            (tau, info, flag, msg) = opt.fsolve(gap, x0, args=(), fprime=None, full_output=True)
            print('> calc tau')
            print('  message: %s after %d iterations.' % (msg[:-1], info['nfev']))
            print('  x0  = %f' % x0[0])
            print('  tau = %f' % tau[0])
            print('  gap = %e' % gap(tau))

        return tau[0]

    def __calc_params__(self, data, method, verbose=False):
        # ToDo: Check if it is better to use args instead of lambda functions
        if verbose:
            print('  Estimating parameters using %s method' % method)

        # start point (z)
        beta = 6.748686e-02
        gamma = 1.457889e+00
        theta = 3.686879e+01
        z = np.array([beta, gamma, theta])

        if verbose:
            gap = self.__gap_params__(data, z[0], z[1], z[2])
            print('> gap init')
            print('  beta  = %e [gap = % e]' % (z[0], gap[0]))
            print('  gamma = %e [gap = % e]' % (z[1], gap[1]))
            print('  theta = %e [gap = % e]' % (z[2], gap[2]))

        # solve gap equations
        if method is 'fsolve':
            noneq = lambda z: self.__gap_params__(data, z[0], z[1], z[2])
            try:
                (z, info, flag, msg) = opt.fsolve(noneq, z, full_output=True)
                if flag is not 1:
                    msg = 'fsolve failed after %d iterations' % (info['nfev'])
                else:
                    msg = 'fsolve converged after %d iterations' % (info['nfev'])
            except:
                msg = '\033[93m fsolve has raised an exception\033[0m'
                print msg
                z = np.array([beta, gamma, theta])
        else:
            raise Exception('Unsupported method %s' % (method))

        if verbose:
            gap = self.__gap_params__(data, z[0], z[1], z[2])
            print('> gap final')
            print('  message: %s' % msg)
            print('  beta  = %e [gap = % e]' % (z[0], gap[0]))
            print('  gamma = %e [gap = % e]' % (z[1], gap[1]))
            print('  theta = %e [gap = % e]' % (z[2], gap[2]))
        return z[0], z[1], z[2]

    def __gap_params__(self, data, beta, gamma, theta, verbose=False):
        G1112A = lambda z: mp.meijerg([[1 - 1 / gamma], []], [[0], [-1 / gamma]], (z ** gamma) / theta)
        G1112B = lambda z: mp.meijerg([[0], []], [[0], [1]], (z ** gamma) / theta)
        G1223 = lambda z: mp.meijerg([[0, 1 - 1 / gamma], []], [[0], [-1 / gamma, 1]], (z ** gamma) / theta)
        G1001 = lambda z: mp.meijerg([[], []], [[0], []], (z ** gamma) / theta)
        G1334 = lambda z: mp.meijerg([[0, 1 - 1 / gamma, 1 - 1 / gamma], []], [[0], [-1 / gamma, -1 / gamma, 1]], (z ** gamma) / theta)
        G1445 = lambda z: mp.meijerg([[0, 1 - 1 / gamma, 1 - 1 / gamma, 1 - 1 / gamma], []], [[0], [-1 / gamma, -1 / gamma, -1 / gamma, 1]], (z ** gamma) / theta)

        # gap beta
        si = 0
        for Ti in data.censorTimes:
            si += Ti * (1 - G1112A(Ti) / gamma)
        gap_beta = -si + len(data.allFailures) / beta

        # gap gamma
        si = 0
        for Ti in data.censorTimes:
            si += Ti * (mp.ln(Ti) * G1334(Ti) - G1445(Ti) / gamma)
        sij = 0
        for tij in data.allFailures:
            sij += (mp.ln(tij) * G1112B(tij)) / (1 - G1001(tij))
        gap_gamma = (beta / gamma) * si - sij

        # gap theta
        si = 0
        for Ti in data.censorTimes:
            si += Ti * G1223(Ti)
        sij = 0
        for tij in data.allFailures:
            sij += G1112B(tij) / (1 - G1001(tij))
        gap_theta = (beta * theta / gamma) * si - theta * sij

        if verbose:
            print('  gap beta  = %g' % gap_beta)
            print('  gap gamma = %g' % gap_gamma)
            print('  gap theta = %g' % gap_theta)

        return np.array([gap_beta, gap_gamma, gap_theta], dtype=float)

    def plot(self, axis=plt):
        tmax = np.max(self.data.censorTimes)
        t = np.linspace(0, tmax)
        Nt = self.ExpectedNumberOfFailures(t, self.beta, self.gamma, self.theta)
        axis.plot(t, Nt, label='GA', marker='s')

    def ExpectedNumberOfFailures(self, t, beta=None, gamma=None, theta=None, verbose=False):
        # Calc integral(intensity(s),s=0..t)

        if beta is None:beta = self.beta
        if theta is None:theta = self.theta
        if gamma is None: gamma = self.gamma

        f = lambda t: self.intensity(t, beta, gamma, theta)

        # t is just a number
        if isinstance(t, Number):
            Nt, abserr = integrate.quad(f, 0, t)
            if verbose:
                print('  Estimate of absolute error in quadrature %g' % abserr)
        else:
            Nt = np.zeros(len(t))
            for i in range(len(t)):
                Nt[i], _ = integrate.quad(f, 0, t[i])
        return Nt

    def ExpectedCostPerUnitOfTime(self, beta, gamma, theta, tau, data, verbose=False):
        # Calculates H(t), the expected cost per unit of time
        # See [Gilardoni2007] eq.(2) pp. 49
        H = (data.CPM + data.CMR * self.ExpectedNumberOfFailures(tau, beta, gamma, theta, verbose)) / tau;
        return H

    # BOOTSTRAP ====================================================================
    def __bootstrap__(self, data, method, verbose=False):
        # set estimatives using full data
        (beta, gamma, theta, tau, H) = self.__bootfun__(data, method, verbose);

        # set confidence intervals using bootstrap
        nboot = 0
        [ci, bootstat] = bootci(nboot, self.__bootfun__, data)
        ci = dict(beta=ci[0], gamma=ci[1], theta=ci[2], tau=ci[3], H=ci[4])

        return beta, gamma, theta, tau, H, ci

    def __bootfun__(self, data, method='fsolve', verbose=False):
        # set beta, gamma, theta using a nonlinear solver
        (beta, gamma, theta) = self.__calc_params__(data, method, verbose)

        tau = self.__calc_tau__(beta, gamma, theta, data)
        H = self.ExpectedCostPerUnitOfTime(beta, gamma, theta, tau, data)
        return beta, gamma, theta, tau, H


class RepairableModelGB:
    """ Repairable Model GB
        lamba(u) = beta (1 - (1 + u**gamma / theta)**(-1)
    """
    def __init__(self, data):
        print('# Model: GB     -----------------------------------')
        self.data = data

        (self.beta, self.gamma, self.theta, self.tau, self.H, self.ci) = self.__bootstrap__(data, 'newton_raphson', verbose=True)

        print('beta  ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(self.beta, self.ci['beta'][0], self.ci['beta'][1]))
        print('theta ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(self.theta, self.ci['theta'][0], self.ci['theta'][1]))
        print('gamma ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(self.gamma, self.ci['gamma'][0], self.ci['gamma'][1]))
        print('tau .............. {:9.3g} [{:9.3g}, {:9.3g}] x 1000 hours'.format(self.tau, self.ci['tau'][0], self.ci['tau'][1]))
        print('H ................ {:9.3g} [{:9.3g}, {:9.3g}] / 1000 hours'.format(self.H, self.ci['H'][0], self.ci['H'][1]))

        logl = lnlike(self, data)
        print 'AIC .............. {:9.3g}'.format(AIC(logl, 3))
        print 'AICc ............. {:9.3g}'.format(AICc(logl, 3, data.numberOfSystems))
        print 'BIC .............. {:9.3g}'.format(BIC(logl, 3, data.numberOfSystems))
        print 'MSE .............. {:9.3g}'.format(MSE(self, data))

    def intensity(self, t, beta=None, gamma=None, theta=None):
        if beta is None: beta = self.beta
        if gamma is None: gamma = self.gamma
        if theta is None: theta = self.theta

        return beta * t**gamma / (t**gamma + theta)

    def ExpectedNumberOfFailures(self, t, beta=None, gamma=None, theta=None, verbose=False):
        # Calc integral(intensity(s),s=0..t)
        def f(t):
            return self.intensity(t, beta, gamma, theta)

        # t is just a number
        if isinstance(t, Number):
            Nt, abserr = integrate.quad(f, 0, t)
            if verbose:
                print('  Estimate of absolute error in quadrature %g' % abserr)
        else:
            Nt = np.zeros(len(t))
            for i in range(len(t)):
                Nt[i], _ = integrate.quad(f, 0, t[i])
        return Nt

    def ExpectedCostPerUnitOfTime(self, beta, gamma, theta, tau, data, verbose=False):
        # Calculates H(t), the expected cost per unit of time
        # See [Gilardoni2007] eq.(2) pp. 49
        H = (data.CPM + data.CMR * self.ExpectedNumberOfFailures(tau, beta, gamma, theta, verbose)) / tau
        return H

    def plot(self, axis=plt):
        tmax = np.max(self.data.censorTimes)
        t = np.linspace(0, tmax)
        Nt = self.ExpectedNumberOfFailures(t, self.beta, self.gamma, self.theta)
        axis.plot(t, Nt, label='GB', marker='s')

    def __calc_tau__(self, beta, gamma, theta, data, verbose=False):
        # using a nonlinear solver (fsolve)
        # ToDo(1): Add derivative of gap (fprime)
        # ToDo(2): Check if it is better to use args instead of lambda functions

        kappa = data.CPM / data.CMR
        def G1333(z): return mp.meijerg([[0,1-1/gamma,0], []], [[0], [1,-1/gamma]], (z**gamma) / theta)
        def gap(tau): return kappa + beta * tau[0] * G1333(tau[0])

        # set initial tau
        tau = [5.85]  # from ModelPLP

        if verbose:
            print('> calc tau')
            print('     tau = %f (init)' % tau[0])
            print('     gap = %e (init)' % gap(tau))

        # solve gap(tau) = 0
        tstart = time.time()
        (tau, info, flag, msg) = opt.fsolve(gap, tau, args=(), fprime=None, full_output=True)
        eta = time.time() - tstart
        if verbose:
            print('     message: %s after %d iterations in %3.2f seconds.' % (msg[:-1], info['nfev'], eta))
            print('     tau = %f (final)' % tau[0])
            print('     gap = %e (final)' % gap(tau))

        return tau[0]

    def __calc_params__(self, data, method, verbose=False):
        # ToDo: Check if it is better to use args instead of lambda functions
        # ToDo: Improve the start point
        if verbose:
            print('  Estimating parameters using %s method' % method)

        # start point (z)
        beta  = 9.556746e-02
        gamma = 1.559009e+00
        theta = 5.984159e+01
        z = np.array([beta, gamma, theta])

        # set elapsed time function
        def eta(t): return time.time() - t
        if verbose:
            tstart = time.time()
            gap = self.__gap_params__(data, z[0], z[1], z[2])
            print('> gap init: eval %3.2f seconds' % eta(tstart))
            print('  beta  = %e [gap = % e]' % (z[0], gap[0]))
            print('  gamma = %e [gap = % e]' % (z[1], gap[1]))
            print('  theta = %e [gap = % e]' % (z[2], gap[2]))

        def eta(t): return time.time() - t

        # solve gap equations
        tstart = time.time()
        if method is 'fsolve':
            noneq = lambda z: self.__gap_params__(data, z[0], z[1], z[2])
            try:
                (z, info, flag, msg) = opt.fsolve(noneq, z, full_output=True)
                if flag is not 1:
                    msg = 'fsolve failed after %d iterations' % (info['nfev'])
                else:
                    msg = 'fsolve converged after %d iterations and %3.2f seconds' % (info['nfev'], eta(tstart))
            except:
                msg = '\033[93m fsolve has raised an exception\033[0m'
                print msg
                z = np.array([beta, gamma, theta])
        elif method is 'newton_raphson':
            noneq = lambda z: self.__gap_params__(data, z[0], z[1], z[2])
            try:
                z, niter = newton_raphson(noneq, z)
                msg = 'newton_raphson converged after %d iterations and %3.2f seconds' % (niter, eta(start))
            except:
                msg = '\033[93m newton_raphson has raised an exception\033[0m' + sys.exc_info()[0]
                print msg
                z = np.array([beta, gamma, theta])
        else:
            raise Exception('Unsupported method %s' % (method))

        if verbose:
            gap = self.__gap_params__(data, z[0], z[1], z[2])
            print('> gap final')
            print('  message: %s' % msg)
            print('  beta  = %e [gap = % e]' % (z[0], gap[0]))
            print('  gamma = %e [gap = % e]' % (z[1], gap[1]))
            print('  theta = %e [gap = % e]' % (z[2], gap[2]))
        return z[0], z[1], z[2]

    def __gap_params__(self, data, beta, gamma, theta, verbose=False):
        def G1112A(z): return mp.meijerg([[1-1/gamma,0], []], [[0], [-1/gamma]], (z ** gamma) / theta)
        def G1333(z): return mp.meijerg([[0,1-1/gamma,0], []], [[0], [-1/gamma,1]], (z ** gamma) / theta)
        def G1112B(z): return mp.meijerg([[0, 0], []], [[0], [1]], (z ** gamma) / theta)
        def G1111(z): return mp.meijerg([[0], []], [[0], []], (z ** gamma) / theta)
        def G1444(z): return mp.meijerg([[0,1-1/gamma,1-1/gamma,0], []], [[0], [-1/gamma,-1/gamma,1]], (z ** gamma) / theta)
        def G1555(z): return mp.meijerg([[0,1-1/gamma,1-1/gamma,1-1/gamma,0], []], [[0], [-1/gamma,-1/gamma,-1/gamma,1]], (z ** gamma) / theta)

        Tij = data.allFailures
        Ti = data.censorTimes

        # calculates in advance to save time
        assert isinstance(Tij, np.ndarray)
        GTij = np.zeros(Tij.size, dtype=np.float)
        for k in range(Tij.size):
            GTij[k] = G1112B(Tij[k]) / (1 - G1111(Tij[k]))

        # gap beta
        si = 0
        for ti in Ti:
            si += ti * (1 - G1112A(ti) / gamma)
        gap_beta = -si + Tij.size / beta

        # gap gamma
        si = 0
        for ti in Ti:
            si += ti * (mp.ln(ti) * G1444(ti) - G1555(ti) / gamma)
        sij = np.sum(np.log(Tij) * GTij)
        gap_gamma = (beta / gamma) * si - sij

        # gap theta
        si = 0
        for ti in Ti:
            si += ti * G1333(ti)
        sij = np.sum(GTij)
        gap_theta = (beta * theta / gamma) * si - theta * sij

        if verbose:
            print('  gap beta  = %g' % gap_beta)
            print('  gap gamma = %g' % gap_gamma)
            print('  gap theta = %g' % gap_theta)

        return np.array([gap_beta, gap_gamma, gap_theta], dtype=float)

    # BOOTSTRAP ====================================================================
    def __bootstrap__(self, data, method, verbose=False):
        # set estimatives using full data
        (beta, gamma, theta, tau, H) = self.__bootfun__(data, method, verbose);

        # set confidence intervals using bootstrap
        nboot = 0
        [ci, bootstat] = bootci(nboot, self.__bootfun__, data)
        ci = dict(beta=ci[0], gamma=ci[1], theta=ci[2], tau=ci[3], H=ci[4])

        return beta, gamma, theta, tau, H, ci

    def __bootfun__(self, data, method='fsolve', verbose=False):
        # set beta, gamma, theta using a nonlinear solver
        (beta, gamma, theta) = self.__calc_params__(data, method, verbose)

        tau = self.__calc_tau__(beta, gamma, theta, data)

        H = self.ExpectedCostPerUnitOfTime(beta, gamma, theta, tau, data)
        return beta, gamma, theta, tau, H

if __name__ == '__main__':
    options = dict(PLP=True, GA=True, GB=True, graphics=True, precision=32)

    # set mpmath precision
    mp.prec = options['precision']

    # set instance
    filename = "data/Gilardoni2007.txt"

    # read data
    data = RepairableData(filename)

    # create models
    if options['PLP']:
        modelPLP = RepairableModelPLP(data)
    if options['GA']:
        modelGA = RepairableModelGA(data)
    if options['GB']:
        modelGB = RepairableModelGB(data)

    if options['graphics']:
        # plot data
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        axis = fig.add_subplot(211)
        data.plot_failures(axis)

        # plot models
        axis = fig.add_subplot(212)
        data.plot_mcnf(axis)
        if options['PLP']:
            modelPLP.plot(axis)
        if options['GA']:
            modelGA.plot(axis)
        if options['GB']:
            modelGB.plot(axis)

        axis.legend(loc='upper left')
        # show plot window
        plt.show()