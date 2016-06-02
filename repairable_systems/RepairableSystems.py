# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <markdowncell>

# <a id='index'></a>
# ## Index
# 1. <a href='#packages'>Package import</a>
# 2. <a href='#bootstrap'>Bootstrap</a>
# 3. <a href='#repairable_data'>Repairable Data</a>
# 4. <a href='#model_plp'>Repairable Model PLP</a>
# 5. <a href='#model_pulcini'>Repairable Model Pulcini</a>
# 6. <a href='#model_ga'>Repairable Model GA</a>
# 7. <a href='#running'>Running</a>
# 8. <a href='#references'>References</a>

# <markdowncell>

# ## Theory
# 
# Let $N(t)$ be a Poisson proccess that represents the number of failures in the time interval $(0,t]$. If $N(t)$ 
# has intensity function $\lambda(t)$, then expected number of failures in the time interval $(0,t]$ will be
# 
# $$M(t)=E(N(t))=\int_{0}^t \lambda(u) du$$
# 
# If the cost of minimal repair (MR) has expected cost $CMR$ and the (perfect) preventive maintenance repair costs $CPM$, 
# then for a given intesity function $\lambda(t)$ the optimal preventive maintenance repair should be done after every 
# $\tau$ units of time, where $\tau$ is given by
# 
# $$\frac{CPM}{CMR} = \int_{0}^\tau u \lambda'(u) du$$
# 
# If $n$ independent systems are observed, the likelihood function is 
# 
# $$L(u) = \exp\left(-\sum_{i=1}^{n} M(T_i)\right) \prod_{i,j}\lambda(t_{ij}),$$
# 
# where $t_{ij}$ is the $j$th failure time for the $i$th system and $T_i$ represents (i) the truncate time of time truncated 
# systems or (ii) the time of the last failure when the system is failure truncated.

# <markdowncell>

# ## Packages import  <a id='packages'></a>
# The mpath package is need to call meijerG function (See [1])

# <codecell>

#%matplotlib inline
from mpmath import mp
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.integrate as integrate
from numbers import Number
#import matplotlib

# <markdowncell>

#%% search
def bracket(f,x1,h):
     c  = 1.618033989
     f1 = f(x1)
     x2 = x1 + h; f2 = f(x2)
     # get direction
     if f2 > f1:
          h  = -h
          x2 = x1 + h; f2 = f(x2)
          # check if minimum between x1-h and x1 + h
          if f2 > f1: return x2, x1-h
     # search loop
     for i in range(100):
          h = c * h
          x3 = x2 + h; f3 = f(x3)
          if f3 > f2: return x1,x3
          x1 = x2; x2 = x3
          f1 = f2; f2 = f3
     print 'Bracket did not find a minimum'

def search(f,a,b,tol=1E-03):
     niter = int(-2.078087 * np.log(tol/abs(b-a)))
     R = 0.618033989
     C = 1.0 - R
     # first telescoping
     x1 = R*a + C*b; x2 = C*a + R*b
     f1 = f(x1); f2 = f(x2)
     # main loop
     for i in range(niter):
          if(f1 > f2):
               a = x1
               x1 = x2; f1 = f2
               x2 = C*a + R*b; f2 = f(x2)
          else:
               b = x2
               x2 = x1; f2 = f1
               x1 = R*a + C*b; f1 = f(x1)
     if(f1 < f2): return x1,f1,i
     else: return x2,f2,i

def powell(F,x,h=0.1,tol=1E-06,verbose=False):
     def f(s): return F(x + s * v)   # F in direction of v

     n  = len(x)                     # number of variables
     df = np.zeros(n,dtype=float)    # decreases of F
     u  = np.eye(n,dtype=float) # vectors v are stored by row
     for j in range(30):
          xold = x.copy()
          fold = F(xold)
          print('iter %5d F(x) = % g'%(j, fold))
          
          # first n line searches record decreases of F
          for i in range(n):
               v      = u[i]
               a,b    = bracket(f,0.0,h)
               s,fmin,niter = search(f,a,b)
               df[i]  = fold - fmin
               fold   = fmin
               x      = x + s * v
               print('  search direcion %2d niters = %2d df = % g'%(i,niter,df[i]))
          
          # last line search in the cycle
          v   = x - xold
          a,b = bracket(f,0.0,h)
          s,flast,_ = search(f,a,b)
          x = x + s * v

          # check convergence
          if np.linalg.norm(x - xold)/n < tol:
              return x, j+1, 'powell converged after %d iterations'%(j+1)

          # identify biggest decrease & update v's
          imax = int(np.argmax(df))
          for i in range(imax, n-1):
               u[i]   = u[i+1]
               u[n-1] = v
     raise Exception('Powell method did not converge')

# <a href='#index'>Return to index</a> 
#%% ## Bootstrap <a id='bootstrap'></a>
# Random sampling with replacement (See [2])

# <codecell>

def bootci(nboot, bootfun, data):
     N = data.numberOfSystems
     n = len(bootfun(data))           # number of statistics
     
     # init bootstat matrix
     bootmat = np.zeros((nboot,n))

     # bootstrap kernel
     for k in range(nboot):
          # sampling
          index  = np.random.randint(N,size=N)
          sample = data.sample(index)
          
          # eval bootfun
          bootmat[k,:] = bootfun(sample)

     bootstat = [np.mean(bootmat[:,i]) for i in range(n)]
     ci       = [[np.percentile(bootmat[:,i],2.5),np.percentile(bootmat[:,i],97.5)] for i in range(n)]
     return (ci, bootstat)

# <markdowncell>

# <a href='#index'>Return to index</a> 
# ## Repairable Data <a id='repairable_data'></a>

# <codecell>

class MCNF(object):
     def __init__(this, data):
          # Mean Cumulative Number of Failures (mcnf) as a function of t (time).
          T = data.censorTimes;
          
          # get unique failure times and time interval
          t = [0];
          for i in range(data.numberOfSystems):
               # add censor times
               if T[i] not in t: t.append(T[i])
               for j in range(len(data.failures[i])):
                    tij = data.failures[i][j]
                    # add failure times
                    if tij not in t: t.append(tij)
          t = np.sort(t)          

          # mcnf[j]: mean number of cumulative failures until time t(j)
          # q[j]   : number of uncensored systems until time t(j)
          mcnf = np.zeros(len(t));
          q    = np.zeros(len(t));
          for j in range(len(t)):
               q[j] = np.sum(T > t[j]); 
               for i in range(data.numberOfSystems):
                    if(T[i] > t[j]):
                         mcnf[j] = mcnf[j] + sum(data.failures[i] <= t[j]);          
          
          # adjust the average
          for j in range(len(q)):
               mcnf[j] = mcnf[j] / q[j] if q[j] > 0 else 0               

          this.failures = t;
          this.meanCumulativeNumberOfFailures = mcnf;
          this.numberOfUncensoredSystems = q;
          this.maxCensorTimes = np.max(T);

     def plot(this, axis=plt):
          t    = this.failures;
          mcnf = this.meanCumulativeNumberOfFailures
          x = np.zeros(4 * (len(t)-1));
          y = np.zeros(len(x));
          j = 0;
          for i in range(len(t)-1):          
               x[j] = t[i]
               x[(j+1):(j+4)] = t[i+1]                    
               y[j:(j+3)] = mcnf[i]
               y[j+3] = mcnf[i+1]
               j += 4;
          axis.plot(x,y,'k-')
          axis.set_xlim((0,this.maxCensorTimes * 1.05))
          axis.set_ylim((-0.02,np.max(y) *1.05));
          axis.set_title('Mean Cumulative Number of Failures');
          axis.set_xlabel('Time (x1000 hours)');
          axis.set_ylabel('Number of Failures');          


     def eval(this, t):
         # bracket
         index = 0;
         for i in range(len(this.failures)):
             if(this.failures[i] > t): 
                 index = i - 1; 
                 break
         return this.meanCumulativeNumberOfFailures[index]
          
class RepairableData(object):
     # failures    : list of lists with the time of each failure for each system
     # allFailures : list of each failure
     def __init__(this, filename=None,show=False):
          # create an empty data
          if(filename is None): 
               this.CMR = 0
               this.CPM = 0
               this.numberOfSystems = 0
               this.failures = []
               this.censorTimes = []          
               return

          # read data from file
          with open(filename) as fid:
               values  = np.fromstring(fid.readline(), dtype=np.float,sep=' ') 
               this.CPM = values[0]
               this.CMR = values[1]
               rawdata = fid.readlines()    
               # convert from time from hours to thousand of hours
               times = [np.fromstring(data,dtype=np.float, sep=' ')/1000 for data in rawdata]                            
               this.numberOfSystems = len(times)
               # read failures
               this.failures = [t[0:-1] for t in times]               
               this.allFailures = []
               for fi in this.failures:
                    for fij in fi:
                         this.allFailures.append(fij)
               this.allFailures = np.array(this.allFailures)
               this.numberOfFailures = np.sum([len(failures) for failures in this.failures])                    
               this.censorTimes = np.array([t[-1] for t in times])
               this.__mcnf = MCNF(this)
               if(show): this.show()

     def mcnf(this, t):
          return this.__mcnf.plot(t)

     def plot_mcnf(this, axis=plt):
          this.__mcnf.plot(axis)

     def plot_failures(this, axis=plt):
          for i in range( this.numberOfSystems):
               axis.plot([0, this.censorTimes[i]],[i+1,i+1],'b-')
               axis.plot(this.censorTimes[i],i+1,'yo')
               axis.plot(this.failures[i],[i+1 for j in range(len(this.failures[i]))],'ro')

          axis.set_ylabel('System ID')
          axis.set_xlabel('Failure Times (x1000 hours)') 
          axis.set_ylim(0,this.numberOfSystems + 1)
          axis.set_title('Repairable Data (CMR: {}, CPM: {})'.format(this.CMR,this.CPM))          

     def sample(this,index,show=False):
          data = RepairableData()
          data.CMR = this.CMR
          data.CPM = this.CPM
          data.censorTimes = [this.censorTimes[i] for i in index]
          data.failures = [this.failures[i] for i in index]
          data.numberOfSystems = len(index)
          if(show): data.show()
          return data

# <markdowncell>

# <a href='#index'>Return to index</a> 
# ## Repairable Model PLP <a id='model_plp'></a>
# Power Law Process : $\lambda(t)=\beta t^{\beta-1}/\theta^\beta$

# <codecell>

class RepairableModelPLP(object):
     def __init__(this, data, Algorithm="bootstrap", verbose=True):
          this.data = data;
          if(verbose): print('# Model: PLP --------------------------------------')

          if(Algorithm=="bootstrap"):
               (beta,theta,tau,H,ci) = this.bootstrap(data)

          # set model parameters
          this.beta  = beta;
          this.theta = theta;
          this.tau   = tau;
          this.H     = H;
          this.ci    = ci;

          if(verbose):
               print('beta  ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(beta , ci['beta' ][0], ci['beta' ][1]));
               print('theta ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(theta, ci['theta'][0], ci['theta'][1]));
               print('tau .............. {:9.3g} [{:9.3g}, {:9.3g}] x 1000 hours'.format(tau  , ci['tau'  ][0], ci['tau'  ][1]));
               print('H ................ {:9.3g} [{:9.3g}, {:9.3g}] / 1000 hours'.format(H    , ci['H'    ][0], ci['H'    ][1]));
               # print('L1 distance ...... % 9.3g\n', data.distance(@(t)this.ExpectedNumberOfFailures(t), 'L1'));
               # print('L2 distance ...... % 9.3g\n', data.distance(@(t)this.ExpectedNumberOfFailures(t), 'L2'));

     def ExpectedNumberOfFailures(this,beta,theta,t):
          # Calculates N(t), the expected number of failures until time t.
          # beta  :: plp parameter (optional)
          # theta :: plp parameter (optional)
          # Model evaluation - Expected number of failures until time t.
          Nt = (t/theta) ** beta;
          return Nt

     def ExpectedCostPerUnitOfTime(this, beta, theta, tau, data):
          # Calculates H(t), the expected cost per unit of time
          # See [Gilardoni2007] eq.(2) pp. 49
          H = (data.CPM + data.CMR * this.ExpectedNumberOfFailures(beta,theta,tau)) / tau;
          return H

     def intensity(this,beta,theta,t):
          # Evaluates the intensity function (lambda).
          # t     :: time
          # beta  :: PLP parameter
          # theta :: PLP parameter
          y = (beta/theta) * (t/theta) ** (beta - 1)
          return y
     
     def calc_tau(this, beta, theta, data):
          # See [Gilardoni2007] eq. (5) pp. 50
          tau = theta * (data.CPM / ((beta - 1) * data.CMR)) ** (1/beta);
          return tau

     def gap_tau(this,beta,theta,tau,data):
          # Check the error (gap) in the current tau value.
          # See [Gilardoni2007] eq. (4) pp. 49
          gap = tau * this.intensity(beta,theta,tau) - this.ExpectedNumberOfFailures(beta,theta,tau) - data.CPM/data.CMR;
          return gap

     def plot(this,axis=plt):
          tmax = np.max(this.data.censorTimes)
          t  = np.linspace(0,tmax)
          Nt = this.ExpectedNumberOfFailures(this.beta, this.theta, t)
          axis.plot(t,Nt,label='PLP',marker='d')

     # BOOTSTRAP ====================================================================
     def bootstrap(this, data, verbose=False):
          # set estimatives (full data)
          (beta,theta,tau,H) = this.bootfun(data);          
          
          # calc confidence interval (ci) using bootstrap
          nboot = 10000;
          [ci,bootstat] = bootci(nboot,this.bootfun,data);          
          
          # set confidence intervals
          ci = {'beta':ci[0],'theta':ci[1],'tau':ci[2],'H':ci[3]}

          return (beta,theta,tau,H,ci)

     def bootfun(this, data):
          # set beta and theta using CMLE
          (beta,theta) = this.CMLE(data);

          # set tau = tau(beta,theta)
          tau = this.calc_tau(beta, theta, data);

          # set H(tau) (See eq.(2) [Gilardoni2007] pp. 49)
          H = this.ExpectedCostPerUnitOfTime(beta,theta,tau,data);

          # set output
          return (beta, theta, tau, H)

     # CMLE :: Conditional Maximum Likelihood Estimator =============================
     def CMLE(this,data):
          # See [Ringdon2000] pp. 210 and [Crow1975]
          M = this.CMLE_M(data);
          beta  = this.CMLE_beta(M,data);
          theta = this.CMLE_theta(beta,M,data);
          return (beta,theta)

     def CMLE_M(this,data):
          # See [Ringdon2000] pp. 210
          m = np.zeros(data.numberOfSystems);
          for i in range(data.numberOfSystems):
               ti = data.failures[i];
               Ti = data.censorTimes[i];
               if(len(ti) > 0):
                    m[i] = len(ti) - (ti[-1] == Ti);
               else:
                    m[i] = 0;
          M = sum(m);
          return M

     def CMLE_beta(this,M,data):
          # See [Ringdon2000] pp. 210
          k = 0;
          for i in range(data.numberOfSystems):
               ti = data.failures[i];
               Ti = data.censorTimes[i];
               k  = k + np.sum(np.log(Ti/ti));
          beta = M / k;
          return beta

     def CMLE_theta(this,beta,M,data):
          # See [Ringdon2000] pp. 210
          T     = data.censorTimes;
          theta = sum(T**beta / M)**(1/beta);
          return theta

# <markdowncell>

# <a href='#index'>Return to index</a> 
# ## Repairable Model Pulcini <a id='model_pulcini'></a>
# $\lambda(t) = \beta (1-e^{-t/\theta})$

# <codecell>

class RepairableModelPulcini:
     def __init__(this, data, Algorithm="bootstrap", verbose=True):
          if(verbose): print('# Model: Pulcini ----------------------------------')
          this.data = data;

          #if(Algorithm=="bootstrap"):
          #     (beta,theta,tau,H,ci) = this.bootstrap(data)          

          print(this.gap_theta(8964, data))
          theta = 8964  # this.calc_theta(data,verbose=True) # 8964
          beta  = this.calc_beta(8964, data) # 28.95
          tau   = this.calc_tau(28.95, 8964, data, verbose=False)
          H     = this.ExpectedCostPerUnitOfTime(beta, theta, tau, data)
          ci    = {'beta':(0,0),'theta':(0,0),'tau':(0,0),'H':(0,0)}
                    
          # # set model parameters
          this.beta  = beta;
          this.theta = theta;
          this.tau   = tau;
          this.H     = H;
          this.ci    = ci;

          if(verbose):               
               print('beta  ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(beta , ci['beta' ][0], ci['beta' ][1]));
               print('theta ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(theta, ci['theta'][0], ci['theta'][1]));
               print('tau .............. {:9.3g} [{:9.3g}, {:9.3g}] x 1000 hours'.format(tau  , ci['tau'  ][0], ci['tau'  ][1]));
               print('H ................ {:9.3g} [{:9.3g}, {:9.3g}] / 1000 hours'.format(H    , ci['H'    ][0], ci['H'    ][1]));
               # print('L1 distance ...... % 9.3g\n', data.distance(@(t)this.ExpectedNumberOfFailures(t), 'L1'));
               # print('L2 distance ...... % 9.3g\n', data.distance(@(t)this.ExpectedNumberOfFailures(t), 'L2'));

     def ExpectedNumberOfFailures(this,beta,theta,t):
          # Calculates N(t), the expected number of failures until time t.
          # beta  :: parameter (optional)
          # theta :: parameter (optional)
          # Model evaluation - Expected number of failures until time t.
          Nt = beta * (theta * (np.exp(-t/theta)-1) + t)
          return Nt

     def ExpectedCostPerUnitOfTime(this, beta, theta, tau, data):
          # Calculates H(t), the expected cost per unit of time
          # See [Gilardoni2007] eq.(2) pp. 49
          H = (data.CPM + data.CMR * this.ExpectedNumberOfFailures(beta,theta,tau)) / tau;
          return H

     def intensity(this,beta,theta,t):
          # Evaluates the intensity function (lambda).
          # t     :: time
          # beta  :: PLP parameter
          # theta :: PLP parameter
          y = beta * (1 - np.exp(-t / theta))
          return y
     
     def calc_tau(this, beta, theta, data, verbose=False):
         # Solves the nonlinear equation 
         # int(u * rho'(u), u) - CPM/CMR = 0
         gap = lambda tau: beta * (theta - np.exp (-tau / theta) * (theta + tau)) - data.CPM/data.CMR
         dgp = lambda tau: (beta * tau)/(np.exp(tau/theta) * theta)
         x0  = 6.5 # from PLP Model         
         tau = opt.fsolve(gap,x0,args=(),fprime=None,full_output=False)
         if(verbose):
             (tau,info,flag,msg) = opt.fsolve(gap,x0,args=(),fprime=dgp,full_output=True)
             print('> calc tau')
             print('  message: %s after %d iterations.' %(msg[:-1],info['nfev']))
             print('  x0  = %f'%(x0))
             print('  tau = %f'%(tau[0]))
             print('  gap = %e'%(gap(tau)))
             
         return tau[0]

     def calc_beta(this, theta, data):
          n = data.numberOfSystems
          m = data.numberOfFailures
          T = data.censorTimes     
          s = np.sum(T - theta * (1 - np.exp(-T / theta)))
          return (n * m) / s

     def gap_theta(this, theta, data):
          T = data.censorTimes
          F = data.allFailures
          X = np.exp(F/theta)
          beta = this.calc_beta(theta, data)
          gap  = beta * np.sum(1-np.exp(T/theta) * (1+T/theta)) - np.sum(F / (theta**2 * (X - 1)))
          return gap

     def calc_theta(this, data, verbose=False):
          # using nonlinear solver (fsolve)
          gap = lambda theta: this.gap_theta(theta[0], data)
          x0  = 8964 # from PLP 

          if(verbose):
             (theta,info,flag,msg) = opt.fsolve(gap,x0,args=(),fprime=None,full_output=True)
             print('> calc theta')
             print('  message: %s after %d iterations.' %(msg[:-1],info['nfev']))
             print('  x0    = %f'%(x0))
             print('  theta = %f'%(theta[0]))
             print('  gap   = %e'%(gap(theta)))
          else:
             theta = opt.fsolve(gap,x0,args=(),fprime=None,full_output=False)

          return theta[0]

     def plot(this,axis=plt):
          tmax = np.max(this.data.censorTimes)
          t  = np.linspace(0,tmax)
          Nt = this.ExpectedNumberOfFailures(this.beta, this.theta, t)
          axis.plot(t,Nt,label='Pulcini',marker='o')

     # BOOTSTRAP ====================================================================
     def bootstrap(this, data, verbose=False):
          # set estimatives (full data)
          (beta,theta,tau,H) = this.bootfun(data);          
          
          # calc confidence interval (ci) using bootstrap
          nboot = 10000;
          [ci,bootstat] = bootci(nboot,this.bootfun,data);          
          
          # set confidence intervals
          ci = {'beta':ci[0],'theta':ci[1],'tau':ci[2],'H':ci[3]}

          return (beta,theta,tau,H,ci)

     def bootfun(this, data):
          # set beta and theta
          (beta,theta) = this.calc_params(data);

          # set tau = tau(beta,theta)
          tau = this.calc_tau(beta, theta, data);

          # set H(tau) (See eq.(2) [Gilardoni2007] pp. 49)
          H = this.ExpectedCostPerUnitOfTime(beta,theta,tau,data);

          # set output
          return (beta, theta, tau, H)

     def calc_params(this,data):
         # calc theta
         gap   = lambda theta: this.gap_theta(data, theta)
         x0    = 595.8 # From [Pulcini2001]
         theta = opt.fsolve(gap,x0,args=(),fprime=None)
         
         # calc beta given theta
         n     = 0
         T     = 0
         beta  = n / T - theta * (1 - np.exp(-T/theta))
         return (beta, theta)    
        

# <markdowncell>

# <a href='#index'>Return to index</a> 
# ## Repairable Model GA <a id='model_ga'></a>
# See [?]

# <codecell>

class RepairableModelGA:
     def __init__(this, data,Algorithm="bootstrap",verbose=True):
          if(verbose): print('# Model: GA     ----------------------------------')
          this.data = data
          
          if(Algorithm=="bootstrap"):
              (beta,gamma,theta,tau,H,ci) = this.bootstrap(data,'powell',verbose=True)

          this.beta  = beta
          this.gamma = gamma
          this.theta = theta
          this.tau   = tau
          this.H     = H
          this.ci    = ci

          if(verbose):
               print('beta  ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(beta , ci['beta' ][0], ci['beta' ][1]));
               print('theta ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(theta, ci['theta'][0], ci['theta'][1]));
               print('gamma ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(gamma, ci['gamma'][0], ci['gamma'][1]));
               print('tau .............. {:9.3g} [{:9.3g}, {:9.3g}] x 1000 hours'.format(tau  , ci['tau'  ][0], ci['tau'  ][1]));
               print('H ................ {:9.3g} [{:9.3g}, {:9.3g}] / 1000 hours'.format(H    , ci['H'    ][0], ci['H'    ][1]));               # print('L1 distance ...... % 9.3g\n', data.distance(@(t)this.ExpectedNumberOfFailures(t), 'L1'));
               # print('L2 distance ...... % 9.3g\n', data.distance(@(t)this.ExpectedNumberOfFailures(t), 'L2'));        
    
     def intensity(this,beta,gamma,theta,t):
          return beta * (1 - np.exp(-t**gamma/theta))

     def calc_tau(this,beta,gamma,theta,data,verbose=False):
          # using a nonlinear solver (fsolve)
          # ToDo(1): Add derivative of gap (fprime)
          # ToDo(2): Check if it is better to use args instead of lambda functions 
     
          kappa = data.CPM / data.CMR
          G1223 = lambda z: mp.meijerg([[0,1-1/gamma],[]],[[0],[1,-1/gamma]],(z**gamma)/theta)
          gap   = lambda tau: kappa + beta * tau[0] * G1223(tau[0]) 
          x0    = [6.5]; # from ModelPLP 
          tau   = opt.fsolve(gap,x0,args=(),fprime=None)
                              
          if(verbose):
               (tau,info,flag,msg) = opt.fsolve(gap,x0,args=(),fprime=None,full_output=True)
               print('> calc tau')
               print('  message: %s after %d iterations.' %(msg[:-1],info['nfev']))
               print('  x0  = %f'%(x0[0]))
               print('  tau = %f'%(tau[0]))
               print('  gap = %e'%(gap(tau)))          
          
          return tau[0]     

     def calc_params(this, data, method, verbose=False):
          # ToDo: Check if it is better to use args instead of lambda functions      
          if(verbose):
              print('  Estimating parameters using %s method' % (method))
                       
          # start point (z) from curve fit using PLP and Matlab  
          beta  = 0.06748684
          gamma = 1.457889
          theta =  36.8688
          
          z = np.array([beta, gamma, theta])

          if(verbose):
              print '> Initial gaps'
              this.gap_params(data, z[0], z[1], z[2], verbose=True)
          
          # solve gap equations
          if method is 'fsolve':
              noneq = lambda z: this.gap_params(data, z[0], z[1], z[2])
              (z,info,flag,msg) = opt.fsolve(noneq, z, full_output=True)
              if(flag is not 1): msg = 'fsolve failed after %d iterations'%(info['nfev'])
              else: msg = 'fsolve converged after %d iterations'%(info['nfev'])
          elif method is 'powell':
              noneq = lambda z: np.linalg.norm(this.gap_params(data, z[0], z[1], z[2]))
              z,niter,msg = powell(noneq, z, h=.5)
          
          if(verbose):                   
              gap = this.gap_params(data, z[0], z[1], z[2])
              print('> gap params')
              print('  message: %s' %(msg))
              print('  beta  = %8.7g gap -> %g'%(z[0],gap[0]))
              print('  gamma = %8.7g gap -> %g'%(z[1],gap[1]))
              print('  theta = %8.7g gap -> %g'%(z[2],gap[2]))
          return z[0],z[1],z[2]

     def gap_params(this, data, beta, gamma, theta, verbose=False):
          G1112A = lambda z: mp.meijerg([[1-1/gamma],[]],[[0],[-1/gamma]],(z**gamma)/theta)
          G1112B = lambda z: mp.meijerg([[0],[]],[[0],[1]],(z**gamma)/theta)
          G1223  = lambda z: mp.meijerg([[0,1-1/gamma],[]],[[0],[-1/gamma,1]],(z**gamma)/theta)
          G1001  = lambda z: mp.meijerg([[],[]],[[0],[]],(z**gamma)/theta)
          G1334  = lambda z: mp.meijerg([[0,1-1/gamma,1-1/gamma],[]],[[0],[-1/gamma,-1/gamma,1]],(z**gamma)/theta)
          G1445  = lambda z: mp.meijerg([[0,1-1/gamma,1-1/gamma,1-1/gamma],[]],[[0],[-1/gamma,-1/gamma,-1/gamma,1]],(z**gamma)/theta)
          
          # gap beta
          si = 0
          for Ti in data.censorTimes:
              si += Ti * (1 - G1112A(Ti) / gamma)
          gap_beta = -si + len(data.allFailures) / beta
                       
          # gap gamma
          si = 0
          for Ti in data.censorTimes:
              si += Ti * (mp.ln(Ti) * G1334(Ti) - G1445(Ti)/gamma)
          sij = 0
          for tij in data.allFailures:
              sij += (mp.ln(tij) * G1112B(tij))/(1 - G1001(tij))
          gap_gamma = (beta / gamma) * si - sij
          
          # gap theta
          si  = 0
          for Ti in data.censorTimes:
              si += Ti * G1223(Ti)          
          sij = 0
          for tij in data.allFailures:
              sij += G1112B(tij)/(1-G1001(tij))
          gap_theta = (beta * theta / gamma) * si - theta * sij
       
          if(verbose):
              print('  gap beta  = %g' % (gap_beta))
              print('  gap gamma = %g' % (gap_gamma))
              print('  gap theta = %g' % (gap_theta))

          return np.array([gap_beta, gap_gamma, gap_theta])
          

     def plot(this,axis=plt):
          tmax = np.max(this.data.censorTimes)
          t  = np.linspace(0,tmax)
          Nt = this.ExpectedNumberOfFailures(this.beta, this.gamma, this.theta, t)
          axis.plot(t,Nt,label='GA',marker='s')
     
     def ExpectedNumberOfFailures(this,beta,gamma,theta,t,verbose=False):
          # Calc integral(intensity(s),s=0..t) 
          f = lambda t: this.intensity(beta,gamma,theta,t)
          
          # t is just a number
          if(isinstance(t,Number)):
              Nt,abserr = integrate.quad(f,0,t)
              if(verbose): 
                  print('  Estimate of absolute error in quadrature %g' % (abserr))
          else:
              Nt = np.zeros(len(t))
              for i in range(len(t)):
                   Nt[i],_ = integrate.quad(f,0,t[i])
          return Nt

     def ExpectedCostPerUnitOfTime(this, beta, gamma, theta, tau, data,verbose=False):
          # Calculates H(t), the expected cost per unit of time
          # See [Gilardoni2007] eq.(2) pp. 49
          H = (data.CPM + data.CMR * this.ExpectedNumberOfFailures(beta,gamma,theta,tau,verbose)) / tau;
          return H

     # BOOTSTRAP ====================================================================
     def bootstrap(this, data, method, verbose=False):
          # set estimatives using full data
          (beta,gamma,theta,tau,H) = this.bootfun(data,method,verbose);        
                              
          # set confidence intervals using bootstrap
#          nboot = 100;
#          [ci,bootstat] = bootci(nboot,this.bootfun,data);          
#          ci = {'beta':ci[0],'gamma':ci[1],'theta':ci[2],'tau':ci[3],'H':ci[4]}
          ci = {'beta':(0,0),'gamma':(0,0),'theta':(0,0),'tau':(0,0),'H':(0,0)}
          
          return (beta,gamma,theta,tau,H,ci)

     def bootfun(this, data, method='MaxLikelihood', verbose=False):
          # set beta, gamma, theta using a nonlinear solver
          (beta,gamma,theta) = this.calc_params(data,method,verbose)

          tau = this.calc_tau(beta,gamma,theta,data);
          H   = this.ExpectedCostPerUnitOfTime(beta,gamma,theta,tau,data);
          return (beta, gamma, theta, tau, H)

# <markdowncell>

# <a href='#index'>Return to index</a> 
#%% Running <a id='running'></a>

# <codecell>

bPLP     = False
bPulcini = False
bGA      = True
options = {'graphics':False}

# set instance
filename = "data/Gilardoni2007.txt"

# read data
data = RepairableData(filename)

# create modelsif 
if bPLP    : modelPLP = RepairableModelPLP(data)
if bPulcini: modelPulcini = RepairableModelPulcini(data, verbose=True)
if bGA     : modelGA = RepairableModelGA(data, verbose=True)

if(options['graphics']):
    # plot data
    fig  = plt.figure()
    fig.set_size_inches(10,10)
    axis = fig.add_subplot(211)
    data.plot_failures(axis)
    # plot models
    axis = fig.add_subplot(212) 
    data.plot_mcnf(axis)
    if bPLP    : modelPLP.plot(axis)
    if bPulcini: modelPulcini.plot(axis)
    if bGA     : modelGA.plot(axis)
    axis.legend(loc='upper left')   
    # plot
    plt.show()
    
# <markdowncell>

# <a href='#index'>Return to index</a>
# ## References: <a id='references'></a>
# [1] https://mpmath.googlecode.com/svn/trunk/doc/build/functions/hypergeometric.html#meijerghttps://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29
# 
# [2] https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29
# 
# [3] Gilardoni, Gustavo L., and Enrico A. Colosimo. "Optimal maintenance time for repairable systems." Journal of Quality Technology 39.1 (2007): 48-53.
# 
# [4] Pulcini, Gianpaolo. "A bounded intensity process for the reliability of repairable equipment." Journal of Quality Technology 33.4 (2001): 480-492.

