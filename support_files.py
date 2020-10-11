import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import time
import random
import seaborn as sns


PATH_1 = '/content/gdrive/My Drive/Data'
PATH_2 = '/content/gdrive/My Drive/nico_data'

# Functions for loading the data
def load_data(file_path):
  """
  Reads all data files (metadata and signal matrix data) as python dictionary,
  the pkl and csv files must have the same file name.

  Arguments:
    file_path -- {str} -- path to the iq_matrix file and metadata file

  Returns:
    Python dictionary
  """
  pkl = load_pkl_data(file_path)
  meta = load_csv_metadata(file_path)
  data_dictionary = {**meta, **pkl}
  
  for key in data_dictionary.keys():
    data_dictionary[key] = np.array(data_dictionary[key])

  return data_dictionary



def load_pkl_data(file_path):
    """
    Reads pickle file as a python dictionary (only Signal data).

    Arguments:
      file_path -- {str} -- path to pickle iq_matrix file

    Returns:
      Python dictionary
    """
    path = os.path.join(PATH_1, file_path + '.pkl')
    with open(path, 'rb') as data:
        output = pickle.load(data)
    return output
    

def load_csv_metadata(file_path):
    """
    Reads csv as pandas DataFrame (only Metadata).

    Arguments:
      file_path -- {str} -- path to csv metadata file

    Returns:
      Pandas DataFarme
    """
    path = os.path.join(PATH_1, file_path + '.csv')
    with open(path, 'rb') as data:
        output = pd.read_csv(data)
    return output

    # Function for splitting the data to training and validation


# and function for selecting samples of segments from the Auxiliary dataset
def split_train_val(data):
    """
    Split the data to train and validation set.
    The validation set is built from training set segments of
    geolocation_id 1 and 4.
    Use the function only after the training set is complete and preprocessed.

    Arguments:
      data -- {ndarray} -- the data set to split

    Returns:
      iq_sweep_burst ndarray matrices
      target_type vector
      for training and validation sets
    """
    idx = ((data['geolocation_id'] == 4) | (data['geolocation_id'] == 1)) \
          & (data['segment_id'] % 6 == 0)
    training_x = data['iq_sweep_burst'][np.logical_not(idx)]
    training_y = data['target_type'][np.logical_not(idx)]
    validation_x = data['iq_sweep_burst'][idx]
    validation_y = data['target_type'][idx]
    return training_x, training_y, validation_x, validation_y



def personalized_split_train_val_two(data, percentage=0.75):
    from sklearn.model_selection import train_test_split

    zero_val = np.sum(np.where(data['target_type'] == 0, 1, 0))  # total zero labels
    one_val = np.sum(np.where(data['target_type'] == 1, 1, 0))  # total 1 labels

    if zero_val < one_val:
        min_label = 0
        max_label = 1
    elif zero_val > one_val:
        min_label = 1
        max_label = 0

    target_data = np.array(data['target_type'])
    start_of_track_data = np.array(data['start_of_track'])

    zipped_data = list(zip(target_data, start_of_track_data))

    train_dict_one, validation_x_one, validation_y_one = split_data_nico(zipped_data, min_label, data, percentage)
    new_percent = len(validation_y_one)
    train_dict_two, validation_x_two, validation_y_two = split_data_nico(zipped_data, max_label, data, new_percent)

    validation_x = np.append(validation_x_one, validation_x_two, axis=0)
    validation_y = np.append(validation_y_one, validation_y_two)

    train_dict = append_dict(train_dict_one, train_dict_two)

    return train_dict, validation_x, validation_y



def return_n_sampes(dict_data, n):
    zeros_use = random.sample(range(len(dict_data["target_type"])), n)
    return_dict = {}
    for key in dict_data.keys():
        return_dict[key] = dict_data[key][zeros_use]

    return return_dict
    
    
    
def return_based_on_max(dict_data):
    zeros = np.sum(np.where(dict_data["target_type"] == 0, 1,0))
    zeros_indices = np.where(dict_data["target_type"] == 0)[0]

    ones = np.sum(np.where(dict_data["target_type"] == 1, 1, 0))
    ones_indices = np.where(dict_data["target_type"] == 1)[0]

    print(zeros)
    print(ones)
    
    if zeros > ones:
        ceiling = ones
        ceiling_indices = list(ones_indices)
        zeros_use = random.sample(list(zeros_indices), ceiling)
        ceiling_indices.extend(zeros_use)

    else:
        ceiling = zeros
        ceiling_indices = list(zeros_indices)
        ones_use = random.sample(list(ones_indices), ceiling)
        ceiling_indices.extend(ones_use)


    return_dict = {}
    for key in dict_data.keys():
        return_dict[key] = dict_data[key][ceiling_indices]

    return return_dict
    
    
def shuffle_trio(x, y, z):
    all_labels = list(y)
    indices = list(range(len(all_labels)))
    random.shuffle(indices)

    new_x = [x[i] for i in indices]
    new_y = [y[i] for i in indices]
    new_z = [z[i] for i in indices]
    
    return np.array(new_x), np.array(new_y), np.array(new_z)

    
    

def stratified_split_train_val(data):
    indices = np.arange(len(data['target_type']))

    from sklearn.model_selection import train_test_split
    train_inds, val_inds = train_test_split(
        indices,
        stratify=data['target_type'],
        train_size=0.9
    )
    
    train_dict = {}
    for key in data.keys():
        train_dict[key] = data[key][train_inds]
    
    # print(f"If we are here we are ok: {train_dict.keys()}")
    # iq_sweep_burst = data['iq_sweep_burst'][train_inds]
    # target_type = data['target_type'][train_inds]
    
    validation_x = data['iq_sweep_burst'][val_inds]
    validation_y = data['target_type'][val_inds]
    
    validation_x = np.array(validation_x)
    validation_y = np.array(validation_y)
    
    validation_x = validation_x.reshape(list(validation_x.shape)+[1])
    
    return train_dict, validation_x, validation_y


def stratified_K_split_train_val(data):
    indices = np.arange(len(data['target_type']))

    from sklearn.model_selection import train_test_split
    train_inds, val_inds = train_test_split(
        indices,
        stratify=train_df['target_type'],
        train_size=0.9
    )

    training_x = data['iq_sweep_burst'][train_inds]
    training_y = data['target_type'][train_inds]
    validation_x = data['iq_sweep_burst'][val_inds]
    validation_y = data['target_type'][val_inds]
    return training_x, training_y, validation_x, validation_y


def aux_split(data):
    """
    Selects segments from the auxilary set for training set.
    Takes the first 3 segments (or less) from each track.

    Arguments:
      data {dataframe} -- the auxilary data

    Returns:
      The auxilary data for the training
    """
    
    idx = np.bool_(np.zeros(len(data['track_id'])))
    print(idx)
    print(len(idx))
    
    for track in np.unique(data['track_id']):
        idx |= data['segment_id'] == (data['segment_id'][data['track_id'] == track][:3])

    for key in data.keys():
        data[key] = data[key][idx]
    return data

    # The function append_dict is for concatenating the training set


# with the Auxiliary data set segments


def append_dict(dict1, dict2):
    dict3 = {}
    for key in dict1.keys():
        one = dict1[key]
        two = dict2[key]
        dict3[key] = np.concatenate((one, two), axis = 0)
    return dict3


    
def fft(iq, axis=0):
    """
    Computes the log of discrete Fourier Transform (DFT).

    Arguments:
      iq_burst -- {ndarray} -- 'iq_sweep_burst' array
      axis -- {int} -- axis to perform fft in (Default = 0)

    Returns:
      log of DFT on iq_burst array
    """
    iq = np.log(np.abs(np.fft.fft(hann(iq), axis=axis)))
    return iq


def hann(iq, window=None):
    """
    Preformes Hann smoothing of 'iq_sweep_burst'.

    Arguments:
      iq {ndarray} -- 'iq_sweep_burst' array
      window -{range} -- range of hann window indices (Default=None)
               if None the whole column is taken

    Returns:
      Regulazied iq in shape - (window[1] - window[0] - 2, iq.shape[1])
    """
    if window is None:
        window = [0, len(iq)]

    N = window[1] - window[0] - 1
    n = np.arange(window[0], window[1])
    n = n.reshape(len(n), 1)
    hannCol = 0.5 * (1 - np.cos(2 * np.pi * (n / N)))
    return (hannCol * iq[window[0]:window[1]])[1:-1]


def max_value_on_doppler(iq, doppler_burst):
    """
    Set max value on I/Q matrix using doppler burst vector.

    Arguments:
      iq_burst -- {ndarray} -- 'iq_sweep_burst' array
      doppler_burst -- {ndarray} -- 'doppler_burst' array (center of mass)

    Returns:
      I/Q matrix with the max value instead of the original values
      The doppler burst marks the matrix values to change by max value
    """
    iq_max_value = np.max(iq)
    for i in range(iq.shape[1]):
        if doppler_burst[i] >= len(iq):
            continue
        iq[doppler_burst[i], i] = iq_max_value
    return iq


def normalize(iq):
    """
    Calculates normalized values for iq_sweep_burst matrix:
    (vlaue-mean)/std.
    """
    m = iq.mean()
    s = iq.std()
    return (iq - m) / s


def data_preprocess(data, mode=0):
    """
    Preforms data preprocessing.
    Change target_type lables from string to integer:
    'human'  --> 1
    'animal' --> 0

    Arguments:
      data -- {ndarray} -- the data set

    Returns:
      processed data (max values by doppler burst, DFT, normalization)
    """
    
    X = []
    for i in range(len(data['iq_sweep_burst'])):
        if mode == 0:
            iq = fft(data['iq_sweep_burst'][i])
            iq = max_value_on_doppler(iq, data['doppler_burst'][i])
            iq = normalize(iq)
            X.append(iq)
        elif mode == 1:
            image = data['iq_sweep_burst'][i]
            real = image.real
            real = real.reshape(real.shape[0], real.shape[1], 1)
            imag = image.imag
            imag = imag.reshape(imag.shape[0], imag.shape[1], 1)
            split_complex = np.concatenate((real, imag), axis = 2)
            X.append(split_complex)
    data['iq_sweep_burst'] = np.array(X)
    
    
    if 'target_type' in data:
        data['target_type'][data['target_type'] == 'animal'] = 0
        data['target_type'][data['target_type'] == 'human'] = 1
        data['target_type'][data['target_type'] == 'empty'] = 2

    return data



def split_data_nico(zipped_data, label, data, percentage):
    from sklearn.model_selection import train_test_split

    track, train = get_rel_indices(zipped_data, label)

    potential_validation_indices = np.where(track)
    potential_validation_indices = potential_validation_indices[0]
    temp_vec = np.arange(len(potential_validation_indices))

    guaranteed_train_indices = np.where(train)
    guaranteed_train_indices = guaranteed_train_indices[0]

    if percentage >= 1:
        percentage = percentage / len(potential_validation_indices)
        percentage = 1 - percentage

    train_inds, val_inds = train_test_split(
        temp_vec,
        train_size=percentage
    )

    actual_train_indices = potential_validation_indices[train_inds]
    actual_train_indices = list(actual_train_indices)
    actual_train_indices.extend(guaranteed_train_indices)
    print(f"{label} training indices after split: {len(actual_train_indices)}")

    actual_validation_indices = potential_validation_indices[val_inds]
    print(f"{label} validation indices after split: {len(actual_validation_indices)}")

    train_dict = {}
    for key in data.keys():
        train_dict[key] = data[key][actual_train_indices]

    validation_x = np.array(data['iq_sweep_burst'][actual_validation_indices])
    # validation_x = validation_x.reshape(list(validation_x.shape) + [1])

    validation_y = np.array(data['target_type'][actual_validation_indices])

    return train_dict, validation_x, validation_y
    
    
    
    
def get_rel_indices(zipped_data, label):
    zero_truths = []
    zero_else = []
    for (one, two) in zipped_data:
        if one == label and two == 1:
            zero_truths.append(True)
            zero_else.append(False)
        elif one == label and two != 1:
            zero_else.append(True)
            zero_truths.append(False)
        else:
            zero_truths.append(False)
            zero_else.append(False)

    truth_indices = np.where(zero_truths)
    truth_indices = truth_indices[0]
    temp_vec = np.arange(len(truth_indices))

    return zero_truths, zero_else
    

def describe_dict(dict_data, data_name):
    print(f"######  {data_name}  #########################################")

    keys = dict_data.keys()
    
    key = "target_type"
    curr_list = list(dict_data[key])
    print(f"Length: {len(curr_list)}")

    for item in np.unique(curr_list):
        print(f"Total {int(item)} count: {np.sum(np.where(curr_list == item, 1, 0))}")

    key = "sensor_id"
    if key in keys:
        print(f"Unique Devices: {np.unique(dict_data[key])}")

    key = "iq_sweep_burst"
    if key in keys:
        print(f"Shape of Images: {np.array(dict_data[key]).shape}")

    key = "start_of_track"
    if key in keys:
        print(f"Start of Track: {dict_data[key]}")
        print(f"Number of Individual Tracks: {np.sum(dict_data[key])}")

    print("###########################################################")
    return None




def observe_examples(dict_one):
    all_imgs = dict_one["iq_sweep_burst"]

    fig=plt.figure(figsize=(25, 25))
    columns = 8
    rows = 2
    rand_index = np.random.randint(1)
    used_indices = []
    
    for i in range(16):
        while rand_index in used_indices:
                rand_index = random.randint(0, len(all_imgs))

        img = all_imgs[rand_index]
        used_indices.append(rand_index)
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
        
    plt.show()
    


def flip_array(images):
    images = np.flip(images, axis = 2)
    return images



def return_specific(target, full_data):

    ones = np.where(full_data["target_type"] == target)

    one_images = full_data["iq_sweep_burst"]
    one_images = one_images[ones]

    one_sensor_ids = full_data["sensor_id"]
    one_sensor_ids = one_sensor_ids[ones]

    one_images = normalize_with_profiles(one_images, one_sensor_ids)

    if target == 1:
        new_labels = np.ones((1, one_images.shape[0]))
    elif target == 0:
        new_labels = np.zeros((1, one_images.shape[0]))

    return one_images, new_labels



def normalize_with_profiles(images, ids):
    with open('device_profiles.pkl', 'rb') as fp:
        profiles = pickle.load(fp)

    whole_profile = np.zeros_like(images)
    for i, id_num in enumerate(ids):
        whole_profile[i] = profiles[id_num]

    # print(f"Shape of Profile: {whole_profile.shape}")
    #
    # print(f"Shape of Image: {images.shape}")

    return images - whole_profile




def find_track_starts(target, full_data,):
    # Obtaining the relevant track ids and the relevant track ids
    relevant_track_id = full_data["track_id"]
    relevant_segment_id = full_data["segment_id"]

    ones = np.where(full_data["target_type"] == target)
    relevant_track_id = relevant_track_id[ones]
    relevant_segment_id = relevant_segment_id[ones]

    first_segment_ids = []
    last_segment_ids = []
    is_continuous = []

    for u_track in np.unique(relevant_track_id):
        correct_track = np.where(relevant_track_id == u_track)
        target_seg = relevant_segment_id[correct_track]
        is_continuous.extend(target_seg)

        for j, item in enumerate(target_seg):
            if j == 0:
                continue
            if target_seg[j] != target_seg[j - 1] + 1:
                first_segment_ids.append(target_seg[j])
                last_segment_ids.append(target_seg[j-1])

        first_segment_ids.append(target_seg[0])
        last_segment_ids.append(target_seg[-1])

    sorted_continuous = np.sort(is_continuous)
    repeated = 0
    for n in is_continuous:
        if is_continuous.count(n) > 1:
            repeated += 1

    # print(f"Is Continuous: {checkConsecutive(sorted_continuous)}")
    # print(f"Has repeats: {repeated}")
    # print(f"With min of: {np.min(sorted_continuous)}")
    # print(f"Length: {len(is_continuous)}")
    # print(f"Unique Length: {len(np.unique(is_continuous))}")

    first_segment_ids = np.sort(first_segment_ids)
    last_segment_ids = np.sort(last_segment_ids)
    #
    # print(f"All Start Ids: {first_segment_ids}")
    # print(f"Number of Tracks: {len(first_segment_ids)}")
    # print(f"Number of Start Segment IDs: {len(np.unique(first_segment_ids))}")
    # print(f"Total Errors: {count}")

    coordinates = list(zip(first_segment_ids, last_segment_ids))
    describe_track_lengths(coordinates)

    return coordinates
    
    
    
def check_consecutive(l):
    return sorted(l) == list(range(min(l), max(l) + 1))


def describe_track_lengths(coordinates):
    len_1_track = 0
    longer_track = 0

    for (item1, item2) in coordinates:
        if item1 == item2:
            len_1_track += 1
        else:
            longer_track += 1

    # print(f"Track of 1: {len_1_track}")
    # print(f"Longer tracks: {longer_track}")
    
    

def add_track_starts(full_data):
    # Obtaining the relevant track ids and the relevant track ids
    relevant_track_id = full_data["track_id"]
    relevant_segment_id = full_data["segment_id"]

    parallel_feature = np.zeros_like(np.array(relevant_segment_id))

    # print(f"Number of Tracks Raw: {len(np.unique(relevant_track_id))}")

    for u_track in np.unique(relevant_track_id):
        correct_track = np.where(relevant_track_id == u_track)
        target_seg = relevant_segment_id[correct_track]

        insert_index_start = np.where(relevant_segment_id == target_seg[0])
        insert_index_end = np.where(relevant_segment_id == target_seg[-1])
        parallel_feature[insert_index_start] = 1
        parallel_feature[insert_index_end] = 1

    full_data["start_of_track"] = parallel_feature

    return full_data


def extend_target_four(target, factor, full_data):
    count = 0
    start_indices = find_track_starts(target, full_data)

    # Finding the relevant indices
    ones = np.where(full_data["target_type"] == target)
    temp_img_two = np.zeros((int(len(list(ones[0])) * factor), 126, 32))

    # Obtaining the relevant images
    one_images = full_data["iq_sweep_burst"]
    one_images = one_images[ones]

    # Obtaining the relevant track ids
    relevant_segment_id = full_data["segment_id"]
    relevant_segment_id = relevant_segment_id[ones]

    for (start_index, end_index) in start_indices:
        length = end_index - start_index + 1
        temp_img = np.zeros((126, 32 * length))

        for i, seg_id in enumerate(range(start_index, end_index + 1)):
            find_index = np.where(relevant_segment_id == seg_id)
            one_image = one_images[find_index]
            temp_img[:, i * 32:i * 32 + 32] = one_image

        if length == 1 and temp_img.shape[1] == 32:
            temp_img_two[count] = temp_img
            count += 1
        else:
            # print(temp_img.shape)
            for k in range(int((length - 1)*factor + 1)):
                start = int(k * (32 / factor))
                end = start + 32
                temp_three = temp_img[:, start:end]
                # print(f"Temp Three Shape: {temp_three.shape}")
                if temp_three.shape[1] == 32:
                    temp_img_two[count] = temp_three
                    count += 1

    temp_img_two = temp_img_two[:count - 1, :, :]
    
    print(f"Shape of Extended Images: {temp_img_two.shape}")

    if target == 1:
        new_labels = np.ones((1, temp_img_two.shape[0]))
    elif target == 0:
        new_labels = np.zeros((1, temp_img_two.shape[0]))
    elif target == 2:
        new_labels = np.empty([1, temp_img_two.shape[0]])
        new_labels.fill(2)

    return temp_img_two, new_labels
    
    
def extend_target_two(target, factor, full_data):

    # Finding the relevant indices
    ones = np.where(full_data["target_type"] == target)
    temp_img_two = np.zeros((int(len(list(ones[0]))*factor), 126, 32))

    # Obtaining the relevant images
    one_images = full_data["iq_sweep_burst"]
    one_images = one_images[ones]

    # Obtaining the relevant sensors
    relevant_sensor_id = full_data["sensor_id"]
    relevant_sensor_id = relevant_sensor_id[ones]

    # Normalizing the relevant sensors
    # one_images = normalize_with_profiles(one_images, relevant_sensor_id)

    # Obtaining the relevant track ids
    relevant_track_id = full_data["track_id"]
    relevant_track_id = relevant_track_id[ones]

    # Obtaining the relevant track ids
    relevant_segment_id = full_data["segment_id"]
    relevant_segment_id = relevant_segment_id[ones]

    count = 0

    for u_sensor in np.unique(relevant_sensor_id):

        correct_sensors = np.where(relevant_sensor_id == u_sensor)

        rel_img_sen = one_images[correct_sensors]
        rel_track = relevant_track_id[correct_sensors]
        rel_seg = relevant_segment_id[correct_sensors]
        for u_track in np.unique(rel_track):
            correct_track = np.where(rel_track == u_track)
            # print(f"Correct Track Quantity: {correct_track.shape}")
            target_img = rel_img_sen[correct_track]
            target_seg = rel_seg[correct_track]

            temp_img = np.zeros((126, 32*len(list(np.unique(target_seg))))) 

            for i, item in enumerate(np.sort(np.unique(target_seg))):
                which_next = np.where(target_seg == item)
                to_forget = target_img[which_next]
                temp_img[:, i*32:i*32 + 32] = to_forget           

            for k in range(int((temp_img.shape[1] / (32 / factor)) - 1)):
                start = int(k * (32 / factor))
                end = start + 32
                temp_three = temp_img[:, start:end]
                if temp_three.shape[1] != 32:
                    continue
                else:
                    temp_img_two[count] = temp_three
                    count += 1

    temp_img_two = temp_img_two[:count - 1, :, :]

    if target == 1:
        new_labels = np.ones((1, temp_img_two.shape[0]))
    elif target == 0:
        new_labels = np.zeros((1, temp_img_two.shape[0]))
    elif target == 2:
        new_labels = np.empty([1, temp_img_two.shape[0]])
        new_labels.fill(2)
    return temp_img_two, new_labels


def shuffle_pair(x, y):
    all_labels = list(y)
    indices = list(range(len(all_labels)))
    random.shuffle(indices)

    new_x = [x[i] for i in indices]
    new_y = [y[i] for i in indices]

    return new_x, new_y
    

def relative_quantities(data_targets):
    # return the factor increase required for the target variables to
    # exist in equal quantities (returns val1, val2, where val1 is zero vals
    # upfactor and val2 is ones vals upfactor

    ones = np.sum(np.where(data_targets == 1, 1, 0))
    zeros = np.sum(np.where(data_targets == 0, 1, 0))

    if ones == 0 and zeros != 0:
        return 1, 0
    elif zeros == 0 and ones != 0:
        return 0, 1
    else:
        max_val = np.max([zeros, ones])
        if max_val == zeros:
            return 1, int(round(zeros / ones))
        elif max_val == ones:
            return int(round(ones / zeros)), 1


def data_pipeline(data, upfactor_val):
    # Returns extended images and labels that rae shuffled such that there is ~equal number of different labels
    targets = data["target_type"]

    zero_fact, one_fact = relative_quantities(targets)
    # print(f"Zero Fact: {zero_fact}\n One Fact: {one_fact}")

    if zero_fact != 0:
        zeros_images, zeros_labels = extend_target_four(target=0, factor=zero_fact * upfactor_val, full_data=data)
        if one_fact == 0:
            all_images = zeros_images
            all_labels = zeros_labels

    if one_fact != 0:
        ones_images, ones_labels = extend_target_four(target=1, factor=one_fact * upfactor_val, full_data=data)
        if zero_fact == 0:
            all_images = ones_images
            all_labels = ones_labels.reshape(-1, 1)

    if zero_fact != 0 and one_fact != 0:
        all_images = np.append(ones_images, zeros_images, axis=0)
        all_labels = np.append(ones_labels, zeros_labels)

    new_train_x, new_train_y = shuffle_pair(all_images, all_labels)
    new_train_x = np.array(new_train_x)
    # new_train_x = new_train_x.reshape(list(new_train_x.shape) + [1])
    new_train_y = np.array(new_train_y)

    # describe_dict({"target_type": new_train_y, "iq_sweep_burst": new_train_x})

    return new_train_x, new_train_y
    
    
    
def personalized_split_train_val(data, percentage=0.75):
    from sklearn.model_selection import train_test_split

    min_val = np.inf

    zero_val = np.sum(np.where(data['target_type'] == 0, 1, 0))
    one_val = np.sum(np.where(data['target_type'] == 1, 1, 0))

    if zero_val < one_val:
        # min_val = zero_val
        min_label = 0
        max_label = 1
    elif zero_val > one_val:
        # min_val = one_val
        min_label = 1
        max_label = 0
    else:
        print("Equal???")

    target_data = np.array(data['target_type'])

    indices = np.where(target_data == min_label)
    indices = indices[0]
    temp_vec = np.arange(len(indices))


    train_inds, val_inds = train_test_split(
        temp_vec,
        train_size=percentage
    )

    train_inds = indices[train_inds]

    val_inds = indices[val_inds]
    new_percent = len(val_inds)

    # print(f"{min_label} Valid Indices: {len(val_inds)}")
    # print(f"{min_label} Train Indices: {len(train_inds)}")

    train_dict_one = {}
    for key in data.keys():
        train_dict_one[key] = data[key][train_inds]

    validation_x_one = data['iq_sweep_burst'][val_inds]
    validation_y_one = data['target_type'][val_inds]

    validation_x_one = np.array(validation_x_one)
    validation_y_one = np.array(validation_y_one)

    # validation_x_one = validation_x_one.reshape(list(validation_x_one.shape) + [1])

    #######

    indices = np.where(target_data == max_label)
    indices = indices[0]
    temp_vec = np.arange(len(indices))

    new_percent = new_percent / len(indices)
    # print(f"New Percent: {new_percent}")

    train_inds, val_inds = train_test_split(
        temp_vec,
        train_size= (1 - new_percent)/1.5
    )

    train_inds = indices[train_inds]
    val_inds = indices[val_inds]

    # print(f"{max_label} Valid Indices: {len(val_inds)}")
    # print(f"{max_label} Train Indices: {len(train_inds)}")

    train_dict_two = {}
    for key in data.keys():
        train_dict_two[key] = data[key][train_inds]

    validation_x_two = data['iq_sweep_burst'][val_inds]
    validation_y_two = data['target_type'][val_inds]

    validation_x_two = np.array(validation_x_two)
    validation_y_two = np.array(validation_y_two)

    # validation_x_two = validation_x_two.reshape(list(validation_x_two.shape) + [1])

    validation_x = np.append(validation_x_one, validation_x_two, axis=0)
    validation_y = np.append(validation_y_one, validation_y_two)

    train_dict = append_dict(train_dict_one, train_dict_two)

    #print(f"{min_label} training indices after split: {len(train_dict['target_type'])}")
    #print(f"{max_label} validation indices after split: {len(validation_y)}")

    return train_dict, validation_x, validation_y
    
    

def split_images_in_two(dict_data):
    images = dict_data["iq_sweep_burst"]
    extended_images = []
    labels = dict_data["target_type"]
    extended_labels = []
    
    print(f"Before Split Images: {images.shape}")
    print(f"Before Split Labels: {len(labels)}")
    
    for image, label in list(zip(images, labels)):
        extended_images.append(np.array(image)[:, 0:16])
        extended_images.append(np.array(image)[:, 16:])
        extended_labels.append(label)
        extended_labels.append(label)
        
    print(f"After Split Images: {np.array(extended_images).shape}")
    print(f"After Split Labels: {len(extended_labels)}")
    
    return np.array(extended_images), extended_labels
    
        
        
def split_by_value(dict_data):
  zeros = np.where(dict_data["target_type"] == 0)[0]
  ones = np.where(dict_data["target_type"] == 1)[0]
  zero_dict = {}
  one_dict = {}

  for key in dict_data.keys():
    zero_dict[key] = dict_data[key][zeros]
    one_dict[key] = dict_data[key][ones]

  return zero_dict, one_dict



def extend_target_oneval(target, factor, full_data):
    count = 0
    start_indices = find_track_starts(target, full_data)
    segs_in_long_tracks = 0
    len_one_tracks = 0
    # Finding the relevant indices
    temp_img_two = np.zeros((len(full_data["target_type"]) * factor, 126, 32))

    # Obtaining the relevant images
    one_images = full_data["iq_sweep_burst"]
    relevant_segment_id = full_data["segment_id"]

    for (start_index, end_index) in start_indices:
        length = end_index - start_index + 1
        temp_img = np.zeros((126, 32 * length))

        for i, seg_id in enumerate(range(start_index, end_index + 1)):
            find_index = np.where(relevant_segment_id == seg_id)[0]
            one_image = one_images[find_index]
            temp_img[:, i * 32:i * 32 + 32] = one_image

        if length == 1 and temp_img.shape[1] == 32:
            temp_img_two[count] = temp_img
            count += 1
            len_one_tracks += 1

        else:
            for k in range(int((length - 1)*factor + 1)):
                start = int(k * (32 / factor))
                end = start + 32
                temp_three = temp_img[:, start:end]
                if temp_three.shape[1] == 32:
                    temp_img_two[count] = temp_three
                    count += 1
                    segs_in_long_tracks += 1

    temp_img_two = temp_img_two[:count - 1, :, :]
    print(f"Number of Single Tracks: {len_one_tracks}")
    print(f"Number of Segs in Longer Tracks: {segs_in_long_tracks}")
    print(f"Shape of Extended Images: {temp_img_two.shape}")

    if target == 1:
        new_labels = np.ones((1, temp_img_two.shape[0]))
    elif target == 0:
        new_labels = np.zeros((1, temp_img_two.shape[0]))
    
    new_labels = new_labels[0]
    new_labels = new_labels.reshape(1, -1)
    new_labels = new_labels[0]
    new_labels = [int(x) for x in new_labels]
    new_labels = np.array(new_labels)
    return temp_img_two, np.array(new_labels)



def get_indices_subset(target, relevant, train_p, val_p):
  temp = np.intersect1d(target, relevant)
  train_ind = random.sample(list(temp), train_p)                                              # 1000 0s from train_df for training
  val_ind = [index for index in temp if index not in train_ind]
  val_ind = random.sample(list(val_ind), val_p)                                             # 100 0s from train_df for validation
  return train_ind, val_ind
  
  
  
  
def create_indices_split(zeros, ones, train, aux2, aux1):
  train_indices_t, train_indices_v = get_indices_subset(zeros, train, 500, 200)
  train_indices_t_1, train_indices_v_1 = get_indices_subset(ones, train, 200, 150)

  train_indices_t.extend(train_indices_t_1)
  train_indices_v.extend(train_indices_v_1)
  
  aux2_indices_t, aux2_indices_v = get_indices_subset(zeros, aux2, 500, 200)
  aux2_indices_t_1, aux2_indices_v_1 = get_indices_subset(ones, aux2, 500, 133)

  aux2_indices_t.extend(aux2_indices_t_1)
  aux2_indices_v.extend(aux2_indices_v_1)

  aux1_indices_t = random.sample(list(aux1), 300)                                                 
  aux1_indices_v = [aux1_s for aux1_s in aux1 if aux1_s not in aux1_indices_t]  
  aux1_indices_v = random.sample(aux1_indices_v, 133)                                             

  train_indices_t.extend(aux2_indices_t)
  train_indices_t.extend(aux1_indices_t)

  train_indices_v.extend(aux2_indices_v)
  train_indices_v.extend(aux1_indices_v)

  print(f"Training Indices Length: {len(train_indices_t)}")                                 
  print(f"Validation Indices Length: {len(train_indices_v)}")

  return train_indices_t, train_indices_v  
