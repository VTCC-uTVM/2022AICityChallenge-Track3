
# Import everything needed to edit video clips
from moviepy.editor import *
import pandas as pd
import os
def cut_video(clip1, time_start, time_end, path_video):
    # clip1 = VideoFileClip("test_phone.mp4").subclip(5, 18)
    clip1 = clip1.subclip(time_start, time_end)
    # getting width and height of clip 1
    w1 = clip1.w
    h1 = clip1.h
    
    print("Width x Height of clip 1 : ", end = " ")
    print(str(w1) + " x ", str(h1))
    
    print("---------------------------------------")
    
    # resizing video downsize 50 %
    clip2 = clip1.resize((512, 512))

    # getting width and height of clip 1
    w2 = clip2.w
    h2 = clip2.h
    
    print("Width x Height of clip 2 : ", end = " ")
    print(str(w2) + " x ", str(h2))
    
    print("---------------------------------------")
    clip2.write_videofile(path_video)
#create folder data
if not os.path.isdir('data_process'):
    os.makedirs('data_process')
else: 
    print("folder already exists.")
for i in range(18):
    data_dir = 'data_process/{}'.format(str(i))
    CHECK_FOLDER = os.path.isdir(data_dir)
    if not CHECK_FOLDER:
        os.makedirs(data_dir)
    else:
        print(data_dir, "folder already exists.")
    print(i)
for folder_name in os.listdir('2022/A1'):
    # print(folder_name)
    path_folder = '2022/A1/{}'.format(folder_name)
    path_csv = '{}/{}.csv'.format(path_folder, folder_name)
    print(path_folder, path_csv)

    df = pd.read_csv(path_csv)
    print(df.head())
    print(len(df))
    print(df.columns)
    print(df['User ID'][0])
    file_name = ''
    count = 0
    for i in range(len(df)):
        if df['Filename'][i] !=' ':
            print("df['Filename'][i]", type(df['Filename'][i]), df['Filename'][i])
            file_name = df['Filename'][i].replace(' ', '')
            print('file_name', file_name)
            # print(count)
            count = 0
        else: 
            if df['Label/Class ID'][i] == 'NA':
                continue
            try:
                clip = VideoFileClip("{}/{}.MP4".format(path_folder, file_name))
                ftr = [3600,60,1]
                time_start = sum([a*b for a,b in zip(ftr, map(int,df['Start Time'][i].split(':')))])
                time_end = sum([a*b for a,b in zip(ftr, map(int,df['End Time'][i].split(':')))])
                path_video = 'data_process/{}/{}_{}_{}.MP4'.format(df['Label/Class ID'][i], file_name, df['End Time'][i], df['Start Time'][i+1])
                cut_video(clip, time_start, time_end, path_video)
                #Segment Transition as label 0
                time_start = sum([a*b for a,b in zip(ftr, map(int,df['End Time'][i].split(':')))])
                time_end = sum([a*b for a,b in zip(ftr, map(int,df['Start Time'][i+1].split(':')))])            
                print('time_start', time_start, 'time_end', time_end)
                path_video = 'data_process/{}/{}_{}_{}.MP4'.format(0, file_name, df['End Time'][i], df['Start Time'][i+1])
                cut_video(clip, time_start, time_end, path_video)
                count +=1
            except Exception as e:
                print(e)
                continue
            # break
        print(file_name)




