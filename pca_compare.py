import numpy as np
import time
from modshogun import *
from sklearn.decomposition import PCA as sklearnPCA
from memory_profiler import profile

#@profile
def shogun_pca(all_samples):
	start1 = time.time()
	train_features = RealFeatures(all_samples)
	preprocessor = PCA(AUTO)
	preprocessor.set_target_dim(2)
	preprocessor.init(train_features)
	transformed = preprocessor.apply_to_feature_matrix(train_features)
	end1 = time.time()	
	return (start1, end1)

#@profile
def sklearn_pca(all_samples):
	start2 = time.time()
	sklearn_pca = sklearnPCA(n_components=2)
	sklearn_transf = sklearn_pca.fit_transform(all_samples.T)
	end2 = time.time()	
	return (start2, end2)

np.random.seed(1)
D = np.array([10, 100, 500, 1000]) 
N = np.array([100, 1000, 10000])

for i in xrange(D.shape[0]):
	for j in xrange(N.shape[0]):
		# Dataset generation
		mu_vec1 = np.zeros(D[i])
		mu_vec2 = np.ones(D[i])
		cov_mat = np.identity(D[i])
		class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat, N[j]).T	
		class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat, N[j]).T
		all_samples = np.concatenate((class1_sample, class2_sample), axis=1)

		start1, end1 = shogun_pca(all_samples)
		start2, end2 = sklearn_pca(all_samples)
		# For time analysis, uncomment below line
		print '|', D[i], '|', N[j], '|', end1 - start1, '|', end2 - start2



