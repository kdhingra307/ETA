import numpy as np
from glob import glob
from ETA import config
from tqdm import tqdm


files = glob("{}/{}/{}/*.npz".format(config.model.working_dir,
                                         config.data.path_pattern,
                                         config.split_prefix.format("train")))


total_sum  = 0
total_count = 0
variance_sum = 0

for e in tqdm(files):
    e = np.load(e)['x']
    total_sum += np.sum(e)
    total_count += np.prod(e.shape)
    variance_sum += np.sum(e**2)

mean = total_sum/total_count
std = np.sqrt(variance_sum/total_count  - mean**2)
print("Mean, std of one batch {}, {}".format(np.mean(e), np.std(e)))
print("Mean, std of complete data {}, {}".format(mean, std))
