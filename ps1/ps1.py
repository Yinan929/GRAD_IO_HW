# Load dependencies
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', None)
#Read in raw data
df = pd.read_csv('simulated_income_data.csv')
#Transform projection_factor to weight so they sum up to 1
df_sum = df.groupby(['panel_year', 'household_income']).agg({'projection_factor': 'sum'})
temp = df_sum.groupby(level=0).apply(lambda x:x / float(x.sum())).reset_index()
temp.rename(columns={'projection_factor':'weight'},inplace=True)
df = pd.merge(temp,df[['panel_year', 'household_income','projection_factor']],on=['panel_year', 'household_income'])
print(df.head())
# Dict to map nielsen income bins to consecutive integers
d = {3: 0, 4: 1, 6: 2, 8: 3, 10: 4, 11: 5, 13: 6, 15: 7, 16: 8, 17: 9,
		18: 10, 19: 11, 21: 12, 23: 13, 26: 14, 27: 15, 28: 16, 29: 17, 30: 18}


class FitNielsenIncome(object):
	#Constructor
	# alt = 16 bins
	# not alt = 19 bins
	def __init__(self,counts,use_bins,alternate=False):
		#self.weight = weight
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
				self.points = np.array([0, 5000, 8000, 10000, 12000, 15000, 20000,
								25000, 30000, 35000, 40000, 45000, 50000, 60000, 70000, 100000])
			else:
				self.points = np.array([0, 5000, 8000, 10000, 12000, 15000, 20000, 25000, 30000, 35000,
								40000, 45000, 50000, 60000, 70000, 100000, 125000, 150000, 200000])
		elif use_bins == 'M':
			if alternate:
				self.points = np.array([2499.5, 6499.5, 8999.5, 10999.5, 13499.5, 17499.5, 22499.5,
								27499.5, 32499.5, 37499.5, 42499.5, 47499.5, 54999.5, 64999.5, 84999.5, 150000])
			else:
				self.points = np.array([2499.5, 6499.5, 8999.5, 10999.5, 13499.5, 17499.5, 22499.5, 27499.5, 32499.5, 37499.5,
								42499.5, 47499.5, 54999.5, 64999.5, 84999.5, 112499.5,137499.5,174999.5,250000])

		assert counts.shape[1] == len(self.points),"Wrong number of bins provided!"

	def methodOfMoments_y(self):
		self.mean = np.multiply(self.points,self.counts).sum()/self.counts.sum()
		de_mean = np.square(self.points - self.mean)
		self.var = np.multiply(de_mean,self.counts).sum()/self.counts.sum()
		sigma2 = np.log((self.var/self.mean)+1)
		sigma = np.sqrt(sigma2)
		mu = np.log(self.mean) - sigma2/2
		return mu,sigma


###
# Part 1: Method of Moments
###
def estimate_params(df,alt):
	estimates = np.apply_along_axis(fit_income,axis=1,arr=df.values,alt=alt)
	dat = pd.DataFrame(estimates, columns=['mu', 'sig'], index=df.index)
	return dat

def fit_income(a, alt):
	if alt:
		a = a[0:16]
	inc = FitNielsenIncome(a[None, :], use_bins = 'M',alternate=alt)
	res = inc.methodOfMoments_y()
	return res


x3 = df.pivot_table(index=['panel_year'], columns=['household_income'],
					values='projection_factor').fillna(0)
splitA = x3[x3.index.get_level_values(0) >= 2010]
splitB = x3[x3.index.get_level_values(0) < 2010]
# lognormals = pd.concat(
# 		[estimate_params(splitB, alt=False), estimate_params(splitA, alt=True)], axis=0)
print(estimate_params(splitB, alt=False))

