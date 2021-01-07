import numpy as np
from glob import glob
from ETA import config
from tqdm import tqdm


files = glob("{}/{}/{}/*.npz".format(config.model.working_dir,
                                         config.data.path_pattern,
                                         config.data.split_prefix.format("train")))


total_sum  = 0
total_count = 0
variance_sum = 0

for e in tqdm(files):
    e = np.load(e)['x']
    total_sum += np.sum(e, axis=(0, 1))
    total_count += np.prod(e.shape[:-1])
    variance_sum += np.sum(e**2, axis=(0, 1))

print(variance_sum, total_sum)
mean = total_sum/total_count
std = np.sqrt(variance_sum/total_count  - mean**2)
print("Mean, std of one batch {}, {}".format(np.mean(e, axis=(0, 1)), np.std(e, axis=(0, 1))))
print("Mean, std of complete data {}, {}".format(mean, std))

min_value1 = np.inf
max_value1 = 0
for e in tqdm(files):
    e = (np.load(e)['x'][:, :, 0] - mean[0])/std[0]
    min_v = np.min(e)
    if min_v < min_value1:
        min_value1 = min_v
    
    min_v = np.max(e)
    if min_v > max_value1:
        max_value1 = min_v

print("Min, Max", min_value1, max_value1)

min_value1 = np.inf
max_value1 = 0
for e in tqdm(files):
    e = (np.load(e)['x'][:, :, 1] - mean[1])/std[1]
    min_v = np.min(e)
    if min_v < min_value1:
        min_value1 = min_v
    
    min_v = np.max(e)
    if min_v > max_value1:
        max_value1 = min_v

print("Min, Max", min_value1, max_value1)