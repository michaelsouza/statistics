from mpmath import mp
import numpy as np
import matplotlib.pyplot as plt

# Bootstrap ========================================================================
def bootci(nboot, bootfun, data):
	N = data.numberOfSystems
	n = len(bootfun(data)) 		# number of statistics
	
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

# RepairableData ===================================================================
class MCNF(object):
	def __init__(this, data):
		# Mean Cumulative Number of Failures (mcnf) as a function of t (time).
		T = data.censorTimes;
		
		# get unique failure times and time interval
		t = [0];
		for i in range(data.numberOfSystems):
			# add censor times
			if T[i] not in t: t.append(T[i])
			for j in range(len(data.failureTimes[i])):
				tij = data.failureTimes[i][j]
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
					mcnf[j] = mcnf[j] + sum(data.failureTimes[i] <= t[j]);		
		
		# adjust the average
		for j in range(len(q)):
			mcnf[j] = mcnf[j] / q[j] if q[j] > 0 else 0			

		this.failureTimes = t;
		this.meanCumulativeNumberOfFailures = mcnf;
		this.numberOfUncensoredSystems = q;
		this.maxCensorTimes = np.max(T);

	def plot(this):
		t    = this.failureTimes;
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
		print(x)
		print(y)
		plt.plot(x,y,'k-')
		plt.xlim((0,this.maxCensorTimes * 1.05))
		plt.ylim((-0.02,np.max(y) *1.05));
		plt.title('Mean Cumulative Number of Failures');
		plt.xlabel('time');
		plt.ylabel('Number of Failures');		


	def eval(this, t):
         # bracket
         index = 0;
         for i in range(len(this.failureTimes)):
             if(this.failureTimes[i] > t): 
                 index = i - 1; 
                 break
         return this.meanCumulativeNumberOfFailures[index]
		
class RepairableData(object):
	def __init__(this, filename=None,show=False):
		# create an empty data
		if(filename is None): 
			this.CMR = 0
			this.CPM = 0
			this.numberOfSystems = 0
			this.failureTimes = []
			this.censorTimes = []		
			return

		# read data from file
		with open(filename) as fid:
			values  = np.fromstring(fid.readline(), dtype=np.float,sep=' ') 
			this.CPM = values[0]
			this.CMR = values[1]
			rawdata = fid.readlines()    
			times = [np.fromstring(data,dtype=np.float, sep=' ') for data in rawdata]
			this.numberOfSystems = len(times)
			this.failureTimes = [t[0:-1] for t in times]
			this.censorTimes = [t[-1] for t in times]
			this.__mcnf = MCNF(this)
			if(show): this.show()

	def mcnf(this, t):
		return this.__mcnf.eval(t)

	def plot_mcnf(this):
		this.__mcnf.show()

	def plot_failures(this):
		for i in range( this.numberOfSystems):
			plt.plot([0, this.censorTimes[i]],[i+1,i+1],'b-')
			plt.plot(this.censorTimes[i],i+1,'yo')
			plt.plot(this.failureTimes[i],[i+1 for j in range(len(this.failureTimes[i]))],'ro')

		plt.ylabel('System ID')
		plt.xlabel('Failure Times') 
		plt.ylim(0,this.numberOfSystems + 1)
		plt.title('Repairable Data (CMR: {}, CPM: {})'.format(this.CMR,this.CPM))		

	def sample(this,index,show=False):
		data = RepairableData()
		data.CMR = this.CMR
		data.CPM = this.CPM
		data.censorTimes = [this.censorTimes[i] for i in index]
		data.failureTimes = [this.failureTimes[i] for i in index]
		data.numberOfSystems = len(index)
		if(show): data.show()
		return data

# RepairableModelPLP ===============================================================
class RepairableModelPLP(object):
	def __init__(this, data, Algorithm="bootstrap", verbose=True):
		if(Algorithm=="bootstrap"):
			(beta,theta,tau,H,ci) = this.bootstrap(data)

		tau = this.calc_tau(beta, theta, data)

		if(verbose):
			print('beta  ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(beta , ci['beta' ][0], ci['beta' ][1]));
			print('theta ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(theta, ci['theta'][0], ci['theta'][1]));
			print('tau .............. {:9.3g} [{:9.3g}, {:9.3g}]'.format(tau  , ci['tau'  ][0], ci['tau'  ][1]));
			print('H ................ {:9.3g} [{:9.3g}, {:9.3g}]'.format(H    , ci['H'    ][0], ci['H'    ][1]));
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

	# BOOTSTRAP ====================================================================
	def bootstrap(this, data, verbose=False):
		# set estimatives
		(beta,theta) = this.CMLE(data);		
		tau = this.calc_tau(beta,theta,data);
		H   = this.ExpectedCostPerUnitOfTime(beta,theta,tau,data);     
		
		# call bootstrap
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
			ti = data.failureTimes[i];
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
			ti = data.failureTimes[i];
			Ti = data.censorTimes[i];
			k  = k + np.sum(np.log(Ti/ti));
		beta = M / k;
		return beta

	def CMLE_theta(this,beta,M,data):
		# See [Ringdon2000] pp. 210
		T     = data.censorTimes;
		theta = sum(T**beta / M)**(1/beta);
		return theta


# RepairableModelG =================================================================
class RepairableModelGPulcini:
	def __init__(this, data,Algorithm="bootstrap",verbose=True):       
		if(Algorithm=="bootstrap"):
			(beta,theta,tau,H,ci) = this.bootstrap(data)
            
		tau = this.calc_tau(beta, theta, data)
		if(verbose):
			print('beta  ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(beta , ci['beta' ][0], ci['beta' ][1]));
			print('theta ............ {:9.3g} [{:9.3g}, {:9.3g}]'.format(theta, ci['theta'][0], ci['theta'][1]));
			print('tau .............. {:9.3g} [{:9.3g}, {:9.3g}]'.format(tau  , ci['tau'  ][0], ci['tau'  ][1]));
			print('H ................ {:9.3g} [{:9.3g}, {:9.3g}]'.format(H    , ci['H'    ][0], ci['H'    ][1]));
			# print('L1 distance ...... % 9.3g\n', data.distance(@(t)this.ExpectedNumberOfFailures(t), 'L1'));
			# print('L2 distance ...... % 9.3g\n', data.distance(@(t)this.ExpectedNumberOfFailures(t), 'L2'));        
    
	def intensity(this,beta,gamma,theta):
		return beta * (1 - np.exp(-u**gamma/theta))        
    
	# BOOTSTRAP ====================================================================
	def bootstrap(this, data, verbose=False):
		# set estimatives
		(beta,theta) = this.CMLE(data);
		tau = this.calc_tau(beta,theta,data);
		H   = this.ExpectedCostPerUnitOfTime(beta,theta,tau,data);     
		
		# call bootstrap
		nboot = 10000;
		[ci,bootstat] = bootci(nboot,this.bootfun,data);		
		
		# set confidence intervals
		ci = {'beta':ci[0],'theta':ci[1],'tau':ci[2],'H':ci[3]}

		return (beta,gamma,theta)
    

def plot_solution(model, data):
	data.plot_mcnf()
# Execution ========================================================================
filename = "/home/michael/github/statistics/repairable_systems/data/Gilardoni2007.txt"
data = RepairableData(filename)
# print(data.failureTimes)
# data.plot_failures()
# data.plot_mcnf()
# RepairableModelPLP(data)
# print(mp.meijerg([[0],[]],[[0],[]],.5))

