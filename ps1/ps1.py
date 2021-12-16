# Load dependencies
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.stats import lognorm

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', None)

#Read in raw data
df = pd.read_csv('simulated_income_data.csv')

#Prepare household data for unaggregated data
def prep_agent_data(hh_data,use_projections=True):
	#transform projection factor to weights
	df_sum = df.groupby(['panel_year']).agg({'projection_factor': 'sum'}).reset_index().rename(columns={'projection_factor':'tot'})
	x = hh_data.merge(df_sum,on=['panel_year'],how='left')
	if not use_projections:
		x['projection_factor'] = 1
	else:
		x['projection_factor'] = x['projection_factor']/x['tot']
	return x.groupby(['panel_year', 'household_income'])[['projection_factor']].sum()


# Dict to map nielsen income bins to consecutive integers
d = {3: 0, 4: 1, 6: 2, 8: 3, 10: 4, 11: 5, 13: 6, 15: 7, 16: 8, 17: 9,
		18: 10, 19: 11, 21: 12, 23: 13, 26: 14, 27: 15, 28: 16, 29: 17, 30: 18}


class FitNielsenIncome(object):
	#Constructor
	# alt = 16 bins
	# not alt = 19 bins
	def __init__(self,counts,use_bins,alternate=False):
		self.counts = counts
		if use_bins == 'H':
			if alternate:
				self.points = np.array([4999, 7999, 9999, 11999, 14999, 19999, 24999,
								29999, 34999, 39999, 44999, 49999, 59999, 69999, 99999, 124999])
			else:
				self.points = np.array([4999, 7999, 9999, 11999, 14999, 19999, 24999, 29999, 34999, 39999,
								44999, 49999, 59999, 69999, 99999, 124999, 149999, 199999, 300000])
		elif use_bins == 'L':
			if alternate:
				self.points = np.array([1, 5000, 8000, 10000, 12000, 15000, 20000,
								25000, 30000, 35000, 40000, 45000, 50000, 60000, 70000, 100000])
			else:
				self.points = np.array([1, 5000, 8000, 10000, 12000, 15000, 20000, 25000, 30000, 35000,
								40000, 45000, 50000, 60000, 70000, 100000, 125000, 150000, 200000])
		elif use_bins == 'M':
			if alternate:
				self.points = np.array([2499.5, 6499.5, 8999.5, 10999.5, 13499.5, 17499.5, 22499.5,
								27499.5, 32499.5, 37499.5, 42499.5, 47499.5, 54999.5, 64999.5, 84999.5, 150000])
			else:
				self.points = np.array([2499.5, 6499.5, 8999.5, 10999.5, 13499.5, 17499.5, 22499.5, 27499.5, 32499.5, 37499.5,
								42499.5, 47499.5, 54999.5, 64999.5, 84999.5, 112499.5,137499.5,174999.5,250000])

		assert counts.shape[1] == len(self.points),"Wrong number of bins provided!"

		self.W = np.eye(self.counts.shape[1])

	def methodOfMoments_y(self):
		mean = np.multiply(self.points,self.counts).sum()/self.counts.sum()
		de_mean = np.square(self.points - mean)
		var = np.multiply(de_mean,self.counts).sum()/self.counts.sum()
		sigma = np.sqrt(np.log(var/np.square(mean)+1))
		mu = np.log(np.square(mean)/np.sqrt(var+np.square(mean)))
		return mu,sigma

	def methodOfMoments_lny(self):
		points = np.log(self.points)
		mean = np.multiply(points,self.counts).sum()/self.counts.sum()
		de_mean = np.square(points - mean)
		var = np.multiply(de_mean,self.counts).sum()/self.counts.sum()
		sigma = np.sqrt(var)
		return mean,sigma

	def ML(self):
		points = np.log(self.points)
		mu_hat = np.multiply(points,self.counts).sum()/self.counts.sum()
		sigma_hat2 = np.multiply(np.square(points-mu_hat),self.counts).sum()/self.counts.sum()
		sigma_hat = np.sqrt(sigma_hat2)
		#calculate Hessian
		demean = np.multiply(np.square(points-mu_hat),self.counts).sum()
		n=self.counts.sum()
		H11 = n/sigma_hat2
		H22 = demean/(2*sigma_hat2**3)
		return mu_hat,sigma_hat,H11,H22

	def lognormal_moment(self,theta):
		#transform to log scale and add back np.inf
		right_endpt = np.append(np.log(self.points[:-1]),np.inf)
		left_endpt = np.append(-np.inf,np.log(self.points[:-1]+np.finfo(float).eps))

		(mu, sigma2) = theta
		norm_cdf_diff = stats.norm.cdf(right_endpt, loc=mu, scale=sigma2) - stats.norm.cdf(left_endpt, loc=mu, scale=sigma2)

		empirical_freq = self.counts/self.counts.sum(axis=1)[:, None]
		return empirical_freq - norm_cdf_diff

	def gmm_obj(self,theta):
		q = self.lognormal_moment(theta)
		return q @ self.W @ q.T

	def solve_gmm(self, theta0=(9, 1)):
		return minimize(self.gmm_obj, x0=theta0, method='L-BFGS-B', bounds=[(0,20),(0,10)],options={'gtol': 1e-8, 'disp': False})

	def updateW(self, theta=np.array([0, 0])):
	# if you don't give parameter estimates, estimate first
	# Fix this for empty
		if not np.any(theta > 0):
			res = self.solve_gmm()
			theta = res.x

		# use parameters to get moments and weights
		mom = self.lognormal_moment(theta)
		self.W = np.linalg.pinv(mom.T @ mom)
		return



def estimate_params(df,alt,bins,method):
	estimates = np.apply_along_axis(fit_income,axis=1,arr=df.values,alt=alt,method=method,bins=bins)
	dat = pd.DataFrame(estimates, columns=['mu', 'sig'], index=df.index)
	return dat

def fit_income(a, alt,method,bins):
	if alt:
		a = a[0:16]
	inc = FitNielsenIncome(a[None, :], use_bins = bins,alternate=alt)
	#Estimate mu and sigma working with y
	if method == 'methodOfMoments_y':
		res = inc.methodOfMoments_y()
	#Estimate mu and sigma working with lny
	elif method == 'methodOfMoments_lny':
		res = inc.methodOfMoments_lny()
	elif method =='maximumlikelihood':
		res = inc.ML()[0:2]
	elif method =='gmm':
		#inc.updateW()
		res = inc.solve_gmm().x
	return res


#Partition table for post 2010 and after 2010

x3 = df.pivot_table(index=['panel_year'], columns=['household_income'],
						values='projection_factor').fillna(0)
splitA = x3[x3.index.get_level_values(0) >= 2010]
splitB = x3[x3.index.get_level_values(0) < 2010]

###
# Part 1: Method of Moments
###

def MethodofMoments(splitA,splitB,bins):

	mmy = pd.concat(
			[estimate_params(splitB, alt=False,bins=bins,method ='methodOfMoments_y'), estimate_params(splitA, alt=True,bins=bins,method ='methodOfMoments_y')], axis=0)
	print(mmy)
	mmlny = pd.concat(
			[estimate_params(splitB, alt=False,bins=bins,method ='methodOfMoments_lny'), estimate_params(splitA, alt=True,bins=bins,method ='methodOfMoments_lny')], axis=0)
	print(mmlny)
	return mmlny

#MethodofMoments(splitA,splitB,'H')

###
# Part 2: maximum Likelihood
###


def calHessian(df,alt,bins):
	estimates = np.apply_along_axis(H_entry,axis=1,arr=df.values,alt=alt,bins=bins)
	dat = pd.DataFrame(estimates, columns=['H11','H22'], index=df.index)
	return dat

def H_entry(a, alt,bins):
	if alt:
		a = a[0:16]
	inc = FitNielsenIncome(a[None, :], use_bins = bins,alternate=alt)
	res = inc.ML()[2:]
	return res

def MaximumLikelihood(splitA,splitB,bins):
	ml = pd.concat(
			[estimate_params(splitB, alt=False,bins=bins,method ='maximumlikelihood'), estimate_params(splitA, alt=True,bins=bins,method ='maximumlikelihood')], axis=0)
	print(ml)

	#Estimate the standard errors of MLE using Hessian Matrix
	H = pd.concat(
			[calHessian(splitB, alt=False,bins=bins), calHessian(splitA, alt=True,bins=bins)], axis=0)

	for index,row in H.iterrows():
		h11 = row['H11']
		h22 = row['H22']
		Hessian = np.array([[h11, 0], [0, h22]])
		Hinv = np.linalg.inv(Hessian)
		se = [np.sqrt(Hinv[0][0]),np.sqrt(Hinv[1][1])]
		print(se)
	return ml


#MaximumLikelihood(splitA,splitB,'H')


###
# Part 3: GMM
###
def GMM(splitA,splitB):
	#first stage estimates
	gmm = pd.concat(
			[estimate_params(splitB, alt=False,bins='H',method ='gmm'), estimate_params(splitA, alt=True,bins='H',method ='gmm')], axis=0)
	print(gmm)
	return gmm

#GMM(splitA,splitB)
df_lognormal = GMM(splitA,splitB)

df_lognormal['mean'] = lognorm.mean(df_lognormal['sig'],loc=0, scale=np.exp(df_lognormal['mu']))
df_lognormal['10th'] = lognorm.ppf(0.1,df_lognormal['sig'],loc=0, scale=np.exp(df_lognormal['mu']))
df_lognormal['median'] = lognorm.ppf(0.5,df_lognormal['sig'],loc=0, scale=np.exp(df_lognormal['mu']))
df_lognormal['90th'] = lognorm.ppf(0.9,df_lognormal['sig'],loc=0, scale=np.exp(df_lognormal['mu']))
print(df_lognormal)

