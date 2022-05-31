#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import os
import sys
import pickle
import torch
import random
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter
import cv2
import pandas as pd
import tqdm
"""Wrapper to train and test a video classification model."""
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
import time
logger = logging.get_logger(__name__)
import csv
from itertools import islice
def imresize(im, dsize):
    '''
    Resize the image to the specified square sizes and 
    maintain the original aspect ratio using padding.
    Args:
        im -- input image.
        dsize -- output sizes, can be an integer or a tuple.
    Returns:
        resized image.
    '''
    if type(dsize) is int:
        dsize = (dsize, dsize)
    im_h, im_w, _ = im.shape
    to_w, to_h = dsize
    scale_ratio = min(to_w/im_w, to_h/im_h)
    new_im = cv2.resize(im,(0, 0), 
                        fx=scale_ratio, fy=scale_ratio, 
                        interpolation=cv2.INTER_AREA)
    new_h, new_w, _ = new_im.shape
    padded_im = np.full((to_h, to_w, 3), 128)
    x1 = (to_w-new_w)//2
    x2 = x1 + new_w
    y1 = (to_h-new_h)//2
    y2 = y1 + new_h
    padded_im[y1:y2, x1:x2, :] = new_im 
    # print('padd', padded_im)
    return padded_im
class VideoReader(object):
    def __init__(self, source):
        self.source = source
        try:  # OpenCV needs int to read from webcam
            self.source = int(source)
        except ValueError:
            pass
    def __iter__(self):
        self.cap = cv2.VideoCapture(self.source)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.source))
        return self
    def __next__(self):
        was_read, frame = self.cap.read()
        if not was_read:
            # raise StopIteration
            ## reiterate the video instead of quiting.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = None
            # print('end video')
        return was_read, frame
    def clean(self):
        self.cap.release()
        cv2.destroyAllWindows()
def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x
@torch.no_grad()        
def main(cfg, videoids, labels, path, checkpoint_list):
    """
    Main function to spawn the train and test process.
    """
    print(videoids)
    # print("CFG: ", cfg)
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # print('cfg', cfg.TEST.CHECKPOINT_FILE_PATH)

    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_list[0]
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model.eval()
    print('cfg', cfg.TEST.CHECKPOINT_FILE_PATH)

    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_list[1]
    model_2 = build_model(cfg)
    cu.load_test_checkpoint(cfg, model_2)
    model_2.eval()
    print('cfg', cfg.TEST.CHECKPOINT_FILE_PATH)

    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_list[2]
    model_3 = build_model(cfg)
    cu.load_test_checkpoint(cfg, model_3)
    model_3.eval()
    print('cfg', cfg.TEST.CHECKPOINT_FILE_PATH)

    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_list[3]
    model_4 = build_model(cfg)
    cu.load_test_checkpoint(cfg, model_4)   
    model_4.eval() 
    print('cfg', cfg.TEST.CHECKPOINT_FILE_PATH)

    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_list[4]
    model_5 = build_model(cfg)
    cu.load_test_checkpoint(cfg, model_5)   
    model_5.eval() 
    print('cfg', cfg.TEST.CHECKPOINT_FILE_PATH)

    total_prob_sq={}
    video_order = []
    for key, values in videoids.items():
        video_order.append(values)
        video_path = values[1]
        print(video_path)
        img_provider = VideoReader(video_path)
        fps = 30
        print('fps:', fps)
        frames = []
        s = 0.
        i=-1
        count = 0
        print(cfg.DATA.NUM_FRAMES, cfg.DATA.SAMPLING_RATE)
        predict_sq = []
        prob_sq = []
        score_sq = []
        for able_to_read, frame in img_provider:
            count += 1
            i+=1
            if not able_to_read:
                # when reaches the end frame, clear the buffer and continue to the next one.
                frames = []
                # continue
                break
            if len(frames) != cfg.DATA.NUM_FRAMES and count % cfg.DATA.SAMPLING_RATE ==0:
                frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_processed = cv2.resize(frame_processed, (512, 512), interpolation = cv2.INTER_AREA)
                frames.append(frame_processed)
            if len(frames) == cfg.DATA.NUM_FRAMES:
                start = time.time()
                # Perform color normalization.
                inputs = torch.tensor(np.array(frames)).float()
                inputs = inputs / 255.0
                inputs = inputs - torch.tensor(cfg.DATA.MEAN)
                inputs = inputs / torch.tensor(cfg.DATA.STD)
                # print(cfg.DATA.MEAN, cfg.DATA.STD)
                # 
                # T H W C -> C T H W.
                inputs = inputs.permute(3, 0, 1, 2)
                # 1 C T H W.
                inputs = inputs[None, :, :, :, :]
                # Sample frames for the fast pathway.
                index = torch.linspace(0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES).long()
                fast_pathway = torch.index_select(inputs, 2, index)
                inputs = [inputs]
                # Transfer the data to the current GPU device.
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                # print('inputs[0].shape', inputs[0].shape)
                # Perform the forward pass.
                preds  = model(inputs).detach().cpu().numpy()   
                preds_2  = model_2(inputs).detach().cpu().numpy()   
                preds_3  = model_3(inputs).detach().cpu().numpy()   
                preds_4  = model_4(inputs).detach().cpu().numpy()   
                preds_5  = model_5(inputs).detach().cpu().numpy()   
                prob_ensemble = np.array([preds, preds_2, preds_3, preds_4, preds_5])
                prob_ensemble = np.mean(prob_ensemble, axis=0)
                prob_sq.append(prob_ensemble)
                frames = []
        total_prob_sq[values[0]] = prob_sq
    return dict(sorted(total_prob_sq.items())), video_order
def get_classification(sequence_class_prob):
    classify=[[x,y] for x,y in zip(np.argmax(sequence_class_prob, axis=1),np.max(sequence_class_prob, axis=1))]
    labels_index = np.argmax(sequence_class_prob, axis=1) #returns list of position of max value in each list.
    probs= np.max(sequence_class_prob, axis=1)  # return list of max value in each  list.
    return labels_index,probs
def smoothing(x, k=3):
    ''' Applies a mean filter to an input sequence. The k value specifies the window
    size. window size = 2*k
    '''
    l = len(x)
    s = np.arange(-k, l - k)
    e = np.arange(k, l + k)
    s[s < 0] = 0
    e[e >= l] = l - 1
    y = np.zeros(x.shape)
    for i in range(l):
        y[i] = np.mean(x[s[i]:e[i]], axis=0)
    return y
        
def activity_localization(prob_sq, action_threshold):
    action_idx, action_probs = get_classification(prob_sq)
    print(action_idx)
    print(action_probs)
    threshold = np.mean(action_probs)
    print('threshold:', threshold)
    action_tag = np.zeros(action_idx.shape)
    action_tag[action_probs >=threshold] = 1
    print('action_tag', action_tag)
    activities_idx = []
    startings = []
    endings = []
    for i in range(len(action_tag)):
        if action_tag[i] ==1:
            activities_idx.append(action_idx[i])
            start = i
            end = i+1
            startings.append(start)
            endings.append(end)
    print('activities_idx', activities_idx)  
    print('start', startings)
    print('end', endings)
    return activities_idx,startings,endings

def merge_and_remove(data):
    df_total = pd.DataFrame([[0, 0, 0, 0]], columns=[0, 1, 2, 3])
    print('df_total', df_total)
    for i in range(1, 11):
        # print(i)
        data_video = data[data[0]==i]
        print(data_video)
        list_label = data_video[1].unique()
        print(list_label)
        for label in list_label:
            data_video_label = data_video[data_video[1]== label]
            data_video_label = data_video_label.reset_index()
            print('data_video_label')
            # print(data_video_label)
            # if len(data_video_label) == 1 :
            #     continue
            for j in range(len(data_video_label)-1):
                if data_video_label.loc[j+1, 2] - data_video_label.loc[j, 3] <=16:
                    data_video_label.loc[j+1, 2] = data_video_label.loc[j, 2]
                    data_video_label.loc[j, 3] = 0
                    data_video_label.loc[j, 2] = 0
            print(data_video_label)
            df_total = df_total.append(data_video_label)

            # print('data_video_label[0][1]', data_video_label[0][0])
        
    print('df_total', df_total)

    df_total = df_total[df_total[3]!=0]
    df_total = df_total[df_total[3] - df_total[2] >6]
    df_total = df_total.drop(columns=['index'])
    df_total = df_total.sort_values(by=[0, 1])
    print('df_total', df_total)
    df_total.to_csv('./output/AIC_1404_ensemble_3view_1s_submit.txt', sep=' ', index = False, header=False)
def general_submission(data):
    # data = pd.read_csv(filename, sep=" ", header=None)
    print(data)
    data_filtered = data[data[1] != 0]
    print(data_filtered)
    data_filtered[2] = data[2].map(lambda x: int(float(x)))
    data_filtered[3] = data[3].map(lambda x: int(float(x)))
    data_filtered = data_filtered.sort_values(by=[0,1])
    print(data_filtered)
    # data_filtered.to_csv(r'./output/AIC_1004_ensemble_3view_1s.txt', header=None, index=None, sep=' ', mode='w')
    merge_and_remove(data_filtered)
    # return True
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    fps = 30
    seed_everything(719)
    labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    video_ids={}
    video_names = []
    path = cfg.DATA.PATH_TO_DATA_DIR
    print(path)
    with open('{}/video_ids.csv'.format(path), "r") as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(csvReader):
            if idx > 0:
                video_ids[row[1]] = row[0]
                video_names.append(row[1])
    # path = './data/A2/'
    import glob
    text_files = glob.glob(path + "/**/*.MP4", recursive = True)
    filelist = {}
    for root, dirs, files in os.walk(path):
        for vid_name in files:          # Loop over directories, not files
            if vid_name in video_names:  # Only keep ones that match
                filelist[vid_name] = os.path.join(root, vid_name)
    vid_info = {}
    for key in (video_ids.keys() | filelist.keys()):
        if key in video_ids: vid_info.setdefault(key, []).append(video_ids[key])
        if key in filelist: vid_info.setdefault(key, []).append(filelist[key])
    checkpoint_dashboard_list = ['./checkpoint_submit/checkpoint_epoch_dashboard_24026_00016_77.78.pyth', 
                    './checkpoint_submit/checkpoint_epoch_dashboard_24491_00013_62.86.pyth',
                    './checkpoint_submit/checkpoint_epoch_dashboard_35133_00005_70.59.pyth',
                    './checkpoint_submit/checkpoint_epoch_dashboard_38058_00010_44.12.pyth',
                    './checkpoint_submit/checkpoint_epoch_dashboard_49381_00010_58.33.pyth']  
    vid_info = dict(sorted(vid_info.items()))
    prob_1, video_order = main(cfg, vid_info, labels, filelist, checkpoint_dashboard_list)
    video_ids={}
    video_names = []
    with open('{}/video_ids.csv'.format(path), "r") as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(csvReader):
            if idx > 0:
                video_ids[row[2]] = row[0]
                video_names.append(row[2])
    text_files = glob.glob(path + "/**/*.MP4", recursive = True)
    filelist = {}
    for root, dirs, files in os.walk(path):
        for vid_name in files:          # Loop over directories, not files
            if vid_name in video_names:  # Only keep ones that match
                filelist[vid_name] = os.path.join(root, vid_name)
    vid_info = {}
    for key in (video_ids.keys() | filelist.keys()):
        if key in video_ids: vid_info.setdefault(key, []).append(video_ids[key])
        if key in filelist: vid_info.setdefault(key, []).append(filelist[key])
    vid_info = dict(sorted(vid_info.items()))
    checkpoint_rearview_list = ['./checkpoint_submit/checkpoint_epoch_rearview_24026_00013_69.44.pyth', 
                    './checkpoint_submit/checkpoint_epoch_rearview_24491_00013_60.00.pyth',
                    './checkpoint_submit/checkpoint_epoch_rearview_35133_00013_64.71.pyth',
                    './checkpoint_submit/checkpoint_epoch_rearview_38058_00015_52.94.pyth',
                    './checkpoint_submit/checkpoint_epoch_rearview_49381_00008_61.11.pyth'] 
    prob_2, video_order = main(cfg, vid_info, labels, filelist, checkpoint_rearview_list)
    video_ids={}
    video_names = []
    with open('{}/video_ids.csv'.format(path), "r") as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(csvReader):
            if idx > 0:
                video_ids[row[3]] = row[0]
                video_names.append(row[3])
    text_files = glob.glob(path + "/**/*.MP4", recursive = True)
    filelist = {}
    for root, dirs, files in os.walk(path):
        for vid_name in files:          # Loop over directories, not files
            if vid_name in video_names:  # Only keep ones that match
                filelist[vid_name] = os.path.join(root, vid_name)
    vid_info = {}
    for key in (video_ids.keys() | filelist.keys()):
        if key in video_ids: vid_info.setdefault(key, []).append(video_ids[key])
        if key in filelist: vid_info.setdefault(key, []).append(filelist[key])
    vid_info = dict(sorted(vid_info.items()))
    checkpoint_right_list = ['./checkpoint_submit/checkpoint_epoch_right_24026_00016_80.56.pyth', 
                    './checkpoint_submit/checkpoint_epoch_right_24491_00010_54.29.pyth',
                    './checkpoint_submit/checkpoint_epoch_right_35133_00009_38.24.pyth',
                    './checkpoint_submit/checkpoint_epoch_right_38058_00008_47.06.pyth',
                    './checkpoint_submit/checkpoint_epoch_right_49381_00006_69.44.pyth'] 
    prob_3, video_order = main(cfg, vid_info, labels, filelist, checkpoint_right_list)
    path_file_write = "./output/Output_ensemble_1004_3view_1s.txt"
    text_file = open(path_file_write, "w")
    prob_ensemble = []
    dataframe_list = []
    #ensemble 3 model
    for i in range(1, len(vid_info)+1):
        len_prob = min(len(prob_1[str(i)]), len(prob_2[str(i)]), len(prob_3[str(i)]))
        prob_ensemble_video = []
        for ids in range(len_prob):
            prob_sub_mean = (prob_1[str(i)][ids] + prob_2[str(i)][ids] + prob_3[str(i)][ids])/3
            prob_ensemble_video.append(prob_sub_mean)
        #### post processing
        ###### classification 
        print('post-processing output....')
        prob_actions = np.array(prob_ensemble_video)
        prob_actions = np.squeeze(prob_actions)
        print(prob_actions.shape)
        # ###### temporal localization
        prediction_smoothed = smoothing(prob_actions, 3)
        activity_threshold = 0.4
        activities_idx, startings, endings = activity_localization(prob_actions, activity_threshold)
        print('\Results:')
        print('Video_id\tLabel\tInterval\t\tActivity')
        for idx, s, e in zip(activities_idx, startings, endings):
            start = s * float(cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE) / fps
            end = e * float(cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE) / fps
            label = labels[idx]
            print(
                '{}\t{}\t{:.1f}s - {:.1f}s\t'.format(i, label,start, end))
            # text_file.write('{} {} {} {}\n'.format(i, label,start, end))   
            dataframe_list.append([i, label,start, end])
    # text_file.close()
    data = pd.DataFrame(dataframe_list, columns =[0, 1, 2, 3])
    general_submission(data)
