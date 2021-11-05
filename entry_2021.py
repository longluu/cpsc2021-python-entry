#!/usr/bin/env python3

import numpy as np
import os
import sys
import tensorflow as tf
import wfdb
from utils import qrs_detect, comp_cosEn, save_dict
import metrics

def load_data(sample_path):
    sig, fields = wfdb.rdsamp(sample_path)
    label = fields['comments']

    return sig, label

def challenge_entry(sample_path):
    """
    Load and run model on test set.
    """
    # Load data
    test_sample, label = load_data(sample_path)
    print(label)
    
    # Load pretrained model
    model_load = tf.keras.models.load_model(os.path.join(os.getcwd(), 'trained_model'), custom_objects={'mean_iou': metrics.mean_iou,
     'dice_coefficient': metrics.dice_coefficient})
#     model_load.summary()
    
    # Predict test sample
    label_predict = model_load.predict(tf.convert_to_tensor(np.expand_dims(test_sample, axis=0))).squeeze().round()
    label_predict[np.arange(0, len(label_predict)), np.argmax(label_predict, axis=1)] = 1
    label_predict[label_predict<1] = 0    
    
    # Get the endpoints from predicted label
    end_points = []
    diff_persistentAF = np.diff(np.pad(label_predict[:, 1], (1, 1), 'constant', constant_values=(0, 0)))
    ind_persistentAF_start = np.where(diff_persistentAF==1)[0]
    ind_persistentAF_end = np.where(diff_persistentAF==-1)[0]
    for start, end in zip(ind_persistentAF_start, ind_persistentAF_end):
        end_points.append([int(start), int(end-1)])
        
    diff_paroAF = np.diff(np.pad(label_predict[:, 2], (1, 1), 'constant', constant_values=(0, 0)))
    ind_paroAF_start = np.where(diff_paroAF==1)[0]
    ind_paroAF_end = np.where(diff_paroAF==-1)[0]
    for start, end in zip(ind_paroAF_start, ind_paroAF_end):
        end_points.append([int(start), int(end-1)])
        
    pred_dict = {'predict_endpoints': end_points}
    
    return pred_dict


if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        
    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
            print(sample)
            sample_path = os.path.join(DATA_PATH, sample)
            pred_dict = challenge_entry(sample_path)
            print(pred_dict)

            save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)

