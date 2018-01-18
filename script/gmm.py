import sys
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture

input_path = sys.argv[1]
num_samples = int(sys.argv[2])
num_components = int(sys.argv[3])
output_path = sys.argv[4]

# print('Reading input file...')
inputs = np.load(input_path)
inputs = np.reshape(inputs, (-1, inputs.shape[-1]))
inputs = inputs[~np.all(inputs == 0, axis=1)]  # remove zero row

if num_samples > 0 and num_samples < inputs.shape[0]:
    print('Sampling data...')
    r = np.arange(inputs.shape[0])
    choices = np.random.choice(r, size=num_samples)
    inputs = inputs[choices]

# print('Training GMM...')
gmm = GaussianMixture(num_components, covariance_type='diag')
gmm.fit(inputs)

# print('Writing output...')
with open(f'{output_path}_model.pkl', mode='wb') as f:
    pickle.dump(gmm, f)
np.save(f'{output_path}_priors.npy', gmm.weights_)
np.save(f'{output_path}_means.npy', gmm.means_)
