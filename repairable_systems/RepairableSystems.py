from mpmath import mp
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# Bootstrap ========================================================================
def bootci(nboot, bootfun, data):
	N = data.numberOfSystems
	n = len(bootfun(data)) 		# number of statistics
	
	# init bootstat matrix
	bootmat = np.zeros((nboot,n))

	# bootstrap kernel
	for k in range(0,nboot):
		# sampling
		index  = np.random.randint(N,size=N)
		sample = data.sample(index)
		
		# eval bootfun
		bootmat[k,:] = bootfun(sample)

	bootstat = [np.mean(bootmat[:,i]) for i in range(0,n)]
	ci       = [[np.percentile(bootmat[:,i],2.5),np.percentile(bootmat[:,i],97.5)] for i in range(0,n)]
	return (ci, bootstat)

# RepairableData ===================================================================
class RepairableData(object):
	def __init__(this, filename=None,show=False):
		# create an empty data
		if(filename is None): 
			this.CMR = 0
			this.CPM = 0
			this.numberOfSystems = 0
			this.failureTimes = []
			this.censorTimes = []
		
		# read data from file
		else: 
			with open(filename) as fid:
				values  = np.fromstring(fid.readline(), dtype=np.float,sep=' ') 
				this.CPM = values[0]
				this.CMR = values[1]
				rawdata = fid.readlines()    
				times = [np.fromstring(data,dtype=np.float, sep=' ') for data in rawdata]
				this.numberOfSystems = len(times)
				this.failureTimes = [t[0:-1] for t in times]
				this.censorTimes = [t[-1] for t in times]
				if(show): this.show()

	def show(this):
		for i in range(0, this.numberOfSystems):
			plt.plot([0, this.censorTimes[i]],[i+1,i+1],'b-')
			plt.plot(this.censorTimes[i],i+1,'yo')
			plt.plot(this.failureTimes[i],[i+1 for j in range(0,len(this.failureTimes[i]))],'ro')

		plt.ylabel('System ID')
		plt.xlabel('Failure Times') 
		plt.ylim(0,this.numberOfSystems + 1)
		plt.title('Repairable Data (CMR: {}, CPM: {})'.format(this.CMR,this.CPM))
		plt.show()

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
		# call bootstrap
		nboot = 10000;
		[ci,bootstat] = bootci(nboot,this.bootfun,data);

		# set estimatives
		beta  = bootstat[0];
		theta = bootstat[1];
		tau   = bootstat[2];
		H     = bootstat[3];

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
		for i in range(0,data.numberOfSystems):
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
		for i in range(0,data.numberOfSystems):
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
	
# Execution ========================================================================
filename = "/home/michael/github/statistics/repairable_systems/data/Gilardoni2007.txt"
data = RepairableData(filename)
RepairableModelPLP(data)
# print(mp.meijerg([[0],[]],[[0],[]],.5))

