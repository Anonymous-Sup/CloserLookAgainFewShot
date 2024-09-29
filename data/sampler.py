from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


    
class FewshotSampler(Sampler):
    """
    Randomly sample N classes, and for each class, 
    randomly sample K instances. This process is repeated for `num_episodes` times.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_classes (int): number of classes per episode.
    - num_instances (int): number of instances per class in an episode.
    - num_episodes (int): number of episodes to sample.
    """
    def __init__(self, data_source, num_classes, num_instances, num_episodes):
        
        self.data_source = data_source
        self.num_classes = num_classes  # N classes
        self.num_instances = num_instances  # K instances per class
        self.num_episodes = num_episodes  # Number of episodes

        self.index_dic = defaultdict(list)
        # Create a dictionary mapping each class (pid) to its image indices
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())  # List of unique class IDs (pids)

    def __iter__(self):
        for _ in range(self.num_episodes):  # Iterate over episodes
            episode_idxs = []  # Collect indices for this episode
            
            # Randomly select `num_classes` (N) from available pids (with replacement allowed)
            selected_pids = random.sample(self.pids, self.num_classes)
            
            for pid in selected_pids:
                idxs = self.index_dic[pid]
                
                # Randomly sample `num_instances` (K) from this class
                if len(idxs) >= self.num_instances:
                    selected_idxs = np.random.choice(idxs, size=self.num_instances, replace=False)
                else:
                    # If not enough samples, sample with replacement
                    selected_idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                
                random.shuffle(selected_idxs) 
                episode_idxs.extend(selected_idxs)
            
            # Instead of yielding all indices at once, yield them one by one
            for idx in episode_idxs:
                yield idx

    def __len__(self):
        # estimate number of examples in an epoch
        return self.num_episodes*self.num_classes*self.num_instances
