#!/usr/bin/env python
# coding: utf-8

# In[47]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn
import numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd, sklearn
import librosa, librosa.display
plt.rcParams['figure.figsize'] = (12, 5)


# In[48]:


x, sr = librosa.load('sample.wav', sr= None, offset=0.5 ,duration=2)
ipd.Audio(x, rate=sr)


# In[49]:


spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape


# In[50]:


frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


# In[51]:


spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x, sr=sr)[0]
spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
zero_crossings = librosa.feature.zero_crossing_rate(x)[0]
rms = librosa.feature.rms(x)[0]


# In[55]:


librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r', label = "Spectral Centroid") # normalize for visualization purposes
plt.plot(t, normalize(spectral_bandwidth_2), color='y', label = "Spectral Bandwidth")
plt.plot(t, normalize(spectral_rolloff), color='b', label= "Spectral Roll-Off")
plt.plot(t, normalize(zero_crossings), color='black', label= "Zero Crossing Rate")
plt.plot(t, normalize(zero_crossings), color='green', label= "Root Mean Square Energy")
plt.legend()
plt.xlabel("Time", fontsize=18)
plt.show
plt.savefig('features.eps', format='eps')


# In[7]:


import numpy as np


# In[11]:


plt.plot(np.hanning(2048))
plt.xlabel('n', fontsize=18)
plt.ylabel('w[n]', fontsize=18)
plt.show
plt.savefig('hann.eps', format='eps')


# In[12]:





# In[13]:


Sdb_brahms = librosa.power_to_db(S_brahms)


# In[19]:


librosa.display.specshow(Sdb_brahms, x_axis='time',
                         y_axis='mel', sr=sr, fmax = sr/2)
title='Mel-frequency spectrogram'


# In[24]:


fig, ax = plt.subplots()
S = librosa.feature.melspectrogram(x, sr=sr, power=2.0)
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=sr/2, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
# ax.set(title='Mel-frequency spectrogram',)
plt.xlabel('Time', fontsize=18)
plt.ylabel('Hz', fontsize=18)
plt.savefig('melspec.eps', format='eps')


# In[29]:


mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)


# In[30]:


mfccs.shape


# In[33]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set(title='MFCC')
plt.xlabel('Time', fontsize=18)
plt.savefig('mfcc.eps', format='eps')


# In[36]:


def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def met_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)


def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    
    print("MEL min: {0}".format(fmin_mel))
    print("MEL max: {0}".format(fmax_mel))
    
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = met_to_freq(mels)
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs


# In[37]:


filter_points, mel_freqs = get_filter_points(0, 22050, 26, 2048, sample_rate=44100)


# In[35]:


import numpy as np

def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters


# In[46]:


filters = get_filters(filter_points, 2048)

plt.figure(figsize=(12,5))
for n in range(filters.shape[0]):
    plt.plot(filters[n])
plt.xlabel("Frequency (Hz)", fontsize= 18)
plt.ylabel("Amplitude", fontsize= 18)

plt.savefig("mel_filters.eps", format= "eps")


# In[39]:


filter_points


# In[ ]:




