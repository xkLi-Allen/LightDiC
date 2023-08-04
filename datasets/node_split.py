import json
import torch
import numpy as np


def node_class_split(name, data, split, cache_node_split, official_split, node_split_id,
                    train_size=None, val_size=None, test_size=None, 
                    train_size_per_class=None, val_size_per_class=None, test_size_per_class=None,
                    seed_size=None, seed_size_per_class=None, seed=[], 
                    data_split=10):
    
    if split == "official":
        have_split = ['wikics']
        if name in have_split:
            if name == 'wikics':
                with open(official_split, 'r') as f:
                    ori_data = json.load(f)
                train_masks = torch.tensor(ori_data['train_masks'], dtype=torch.bool)
                train_mask = train_masks[node_split_id]
                train_idx_list = torch.where(train_mask)[0]
                val_masks = torch.tensor(ori_data['val_masks'], dtype=torch.bool)
                val_mask = val_masks[node_split_id]
                val_idx_list = torch.where(val_mask)[0]
                test_mask = torch.tensor(ori_data['test_mask'], dtype=torch.bool)
                test_idx_list = torch.where(test_mask)[0]
                stopping_masks = torch.tensor(ori_data['stopping_masks'], dtype=torch.bool)
                stopping_mask = stopping_masks[node_split_id]
                stopping_idx_list = torch.where(stopping_mask)[0]
                return train_idx_list, val_idx_list, test_idx_list, None, stopping_idx_list
            
        else:
            try:
                cache_node_split_npy = cache_node_split + ".npy"
                split_full = np.load(cache_node_split_npy, allow_pickle=True)
                masks = dict(enumerate(split_full.flatten(), 1))[1]
            except:
                print("Execute node split, it may take a while")
                if train_size is None and train_size_per_class is None:
                    raise ValueError(
                        'Please input the values of train_size or train_size_per_class!')

                if seed_size is not None and seed_size_per_class is not None:
                    raise Warning(
                        'The seed_size_per_class will be considered if both seed_size and seed_size_per_class are given!')
                
                if test_size is not None and test_size_per_class is not None:
                    raise Warning(
                        'The test_size_per_class will be considered if both test_size and test_size_per_class are given!')
                
                if val_size is not None and val_size_per_class is not None:
                    raise Warning(
                        'The val_size_per_class will be considered if both val_size and val_size_per_class are given!')
                
                if train_size is not None and train_size_per_class is not None:
                    raise Warning(
                        'The train_size_per_class will be considered if both train_size and val_size_per_class are given!')

                if len(seed) == 0:
                    seed = list(range(data_split))
                if len(seed) != data_split:
                    raise ValueError(
                        'Please input the random seed list with the same length of {}!'.format(data_split))

                if isinstance(data.y, torch.Tensor):
                    labels = data.y.numpy()
                else:
                    labels = np.array(data.y)
                masks = {}
                masks['train'], masks['val'], masks['test'], masks['seed'] = [], [], [], []
                for i in range(data_split):
                    random_state = np.random.RandomState(seed[i])
                    train_indices, val_indices, test_indices, seed_indices = get_train_val_test_seed_split(random_state,
                                                                                                        labels, train_size_per_class, val_size_per_class, test_size_per_class, seed_size_per_class,
                                                                                                        train_size, val_size, test_size, seed_size)

                    train_mask = np.zeros((labels.shape[0], 1), dtype=int)
                    train_mask[train_indices, 0] = 1
                    val_mask = np.zeros((labels.shape[0], 1), dtype=int)
                    val_mask[val_indices, 0] = 1
                    test_mask = np.zeros((labels.shape[0], 1), dtype=int)
                    test_mask[test_indices, 0] = 1
                    seed_mask = np.zeros((labels.shape[0], 1), dtype=int)
                    if len(seed_indices) > 0:
                        seed_mask[seed_indices, 0] = 1

                    mask = {}
                    mask['train'] = torch.from_numpy(train_mask).bool()
                    mask['val'] = torch.from_numpy(val_mask).bool()
                    mask['test'] = torch.from_numpy(test_mask).bool()
                    mask['seed'] = torch.from_numpy(seed_mask).bool()
                    masks['train'].append(mask['train'])
                    masks['val'].append(mask['val'])
                    masks['test'].append(mask['test'])
                    masks['seed'].append(mask['seed'])
                np.save(cache_node_split, masks)
            train_masks = masks['train']
            val_masks = masks['val']
            test_masks = masks['test']
            seed_masks = masks['seed']
            train_mask = train_masks[node_split_id]
            val_mask = val_masks[node_split_id]
            test_mask = test_masks[node_split_id]
            seed_mask = seed_masks[node_split_id]
            train_idx_list = torch.where(train_mask)[0]
            val_idx_list = torch.where(val_mask)[0]
            test_idx_list = torch.where(test_mask)[0]
            seed_idx_list = torch.where(seed_mask)[0]
            return train_idx_list, val_idx_list, test_idx_list, seed_idx_list, None

    


def sample_per_class(random_state, labels, num_examples_per_class,
                     forbidden_indices=None, force_indices=None):
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    sample_indices_per_class = {index: [] for index in range(num_classes)}
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if ((forbidden_indices is None or sample_index not in forbidden_indices)
                        and (force_indices is None or sample_index in force_indices)):
                    sample_indices_per_class[class_index].append(sample_index)
    if isinstance(num_examples_per_class, int):
        return np.concatenate(
            [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
             for class_index in range(num_classes)
             ])
    elif isinstance(num_examples_per_class, float):
        selection = []
        if force_indices is None:
            values, counts = np.unique(labels, return_counts=True)
        else:
            values, counts = np.unique(
                labels[force_indices], return_counts=True)
        for class_index, count in zip(values, counts):
            size = int(num_examples_per_class*count)
            selection.extend(random_state.choice(
                sample_indices_per_class[class_index], size, replace=False))
        return selection
    else:
        raise TypeError(
            "Please input a float or int number for the parameter num_examples_per_class.")


def get_train_val_test_seed_split(random_state,
                                  labels,
                                  train_size_per_class=None, val_size_per_class=None,
                                  test_size_per_class=None, seed_size_per_class=None,
                                  train_size=None, val_size=None,
                                  test_size=None, seed_size=None):
    num_samples = labels.shape[0]
    remaining_indices = list(range(num_samples))

    if train_size is None and train_size_per_class is None:
        raise ValueError(
            'Please input the values of train_size or train_size_per_class!')
    if seed_size is not None and seed_size_per_class is not None:
        raise Warning(
            'The seed_size_per_class will be considered if both seed_size and seed_size_per_class are given!')
    if test_size is not None and test_size_per_class is not None:
        raise Warning(
            'The test_size_per_class will be considered if both test_size and test_size_per_class are given!')
    if val_size is not None and val_size_per_class is not None:
        raise Warning(
            'The val_size_per_class will be considered if both val_size and val_size_per_class are given!')
    if train_size is not None and train_size_per_class is not None:
        raise Warning(
            'The train_size_per_class will be considered if both train_size and val_size_per_class are given!')

    if train_size_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_size_per_class)
    else:
        if isinstance(train_size, int):
            train_indices = random_state.choice(
                remaining_indices, train_size, replace=False)
        elif isinstance(train_size, float):
            train_indices = random_state.choice(remaining_indices, int(
                train_size*len(remaining_indices)), replace=False)
        else:
            raise TypeError(
                "Please input a float or int number for the parameter train_size.")

    if seed_size_per_class is not None:
        seed_indices = sample_per_class(
            random_state, labels, seed_size_per_class, force_indices=train_indices)
    elif seed_size is not None:
        if isinstance(seed_size, int):
            seed_indices = random_state.choice(
                train_indices, seed_size, replace=False)
        elif isinstance(seed_size, float):
            seed_indices = random_state.choice(train_indices, int(
                seed_size*len(train_indices)), replace=False)
        else:
            raise TypeError(
                "Please input a float or int number for the parameter seed_size.")
    else:
        seed_indices = []

    val_indices = []
    if val_size_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_size_per_class, forbidden_indices=train_indices)
        forbidden_indices = np.concatenate((train_indices, val_indices))
    elif val_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        if isinstance(val_size, int):
            val_indices = random_state.choice(
                remaining_indices, val_size, replace=False)
        elif isinstance(val_size, float):
            val_indices = random_state.choice(remaining_indices, int(
                val_size*len(remaining_indices)), replace=False)
        else:
            raise TypeError(
                "Please input a float or int number for the parameter val_size.")
        forbidden_indices = np.concatenate((train_indices, val_indices))
    else:
        forbidden_indices = train_indices

    if test_size_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_size_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        if isinstance(test_size, int):
            test_indices = random_state.choice(
                remaining_indices, test_size, replace=False)
        elif isinstance(test_size, float):
            test_indices = random_state.choice(remaining_indices, int(
                test_size*len(remaining_indices)), replace=False)
        else:
            raise TypeError(
                "Please input a float or int number for the parameter test_size.")
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    assert len(set(seed_indices)) == len(seed_indices)
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_size_per_class is None:
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_size_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        assert np.unique(train_sum).size == 1

    if val_size_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        assert np.unique(val_sum).size == 1

    if test_size_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices, seed_indices
