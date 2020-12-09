#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
import pandas as pd 
from PIL import Image
import csv
import os 


# In[2]:


classes = os.listdir("../IRMAS-TrainingData/")
header = ['filename', 'class', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'rmse', 'zcr']
for i in range(1, 21):
    header.append('mfcc'+str(i))
for i in range(1, 13):
    header.append('chroma'+str(i))


# In[3]:


df = pd.DataFrame(columns=header)


# In[4]:


i =0
for c in classes:
    for fn in os.listdir(f'../IRMAS-TrainingData/{c}'):
        file = f'../IRMAS-TrainingData/{c}/{fn}'
        audio, sr = librosa.load(file, sr= None, offset=0.5 ,duration=2)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(audio, sr))
        spectral_bandwidth= np.mean(librosa.feature.spectral_bandwidth(audio, sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(audio, sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        rmse = np.mean(librosa.feature.rms(audio))
        chroma_stft = librosa.feature.chroma_stft(audio, sr =sr)
        chroma_stft = np.mean(chroma_stft, axis= 1)
        mfcc  = np.mean(librosa.feature.mfcc(audio, sr=sr), axis=1)
        feats = [file, c, spectral_centroid, spectral_bandwidth, spectral_rolloff, rmse, zcr]
        feats.extend(list(mfcc))
        feats.extend(list(chroma_stft))
        df.loc[i] = feats
        i += 1


# In[6]:


df.to_csv("music_data_all.csv")


# In[7]:


df

