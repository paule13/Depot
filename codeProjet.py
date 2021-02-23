import pandas as pd
import librosa
from librosa import display
import numpy as np
import IPython.display as ipd
from IPython.display import Audio
from glob import glob
import matplotlib.pyplot as plt
from moviepy.editor import *
from pydub import AudioSegment
import scipy as scipy
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
import os
from scipy.spatial import distance

def get_fft(file_path):
   samples, sampling_rate = librosa.load(file_path,sr=None, mono=True, offset=0.0,duration=None )        
   duration = librosa.get_duration(y=samples, sr=sampling_rate)
   fs, data = wavfile.read(file_path)
   data = np.mean(data, axis=1)
   yf=rfft(data)
   return yf



def display_fft(file_path):
   samples, sampling_rate = librosa.load(file_path,sr=None, mono=True, offset=0.0,duration=None )        
   duration = librosa.get_duration(y=samples, sr=sampling_rate)
   fs, data = wavfile.read(file_path)
   data = np.mean(data, axis=1)
   N=duration*sampling_rate
   N=int(N)
   yf=rfft(data)
   yf=yf[:len(yf)//2]
   xf=rfftfreq(N//2,1/sampling_rate)
   xf=np.delete(xf,len(xf)-1)
   plt.plot(xf, np.abs(yf))
   plt.show()
   return yf



def get_time_codes(file_path,coeff):
   samples, sampling_rate = librosa.load(file_path,sr=None, mono=True, offset=0.0,duration=None )        
   duration = librosa.get_duration(y=samples, sr=sampling_rate)
   tempo, beats = librosa.beat.beat_track(y=samples, sr=sampling_rate)
   tab=librosa.frames_to_time(beats, sr=sampling_rate)
   new_tab=[]
   i=0
   if coeff>=1:
       for t in tab:
           if i%coeff==0:
               new_tab.append(t)
           i+=1
   if coeff<=1:
       while i<len(tab)-1:
           new_tab.append(tab[i])
           dist=(tab[i+1]-tab[i])
           invert_coeff=int(1/coeff)
           for k in range (1,invert_coeff):
               new_tab.append(tab[i]+dist*coeff*k)
           i+=1  
   return new_tab


def get_all_fft(file_path,coeff):
    time_codes=get_time_codes(file_path, coeff)
    fft_tab=[]
    min_len_fft=23000
    for i in range (1,len(time_codes)):
        newAudio = AudioSegment.from_wav(file_path)
        newAudio = newAudio[time_codes[i-1]*1000:time_codes[i]*1000]
        newAudio.export('extrait.wav', format="wav")
        fft=get_fft('./extrait.wav')
        if i==1:    #On cherche le fft de longueur minimale
            min_len_fft=len(fft)
        else:
            if min_len_fft>len(fft):
                min_len_fft=len(fft)
        fft_tab.append(fft)     #On retire à tous les fft leurs derniers éléments afin qu'ils aient tous la même longueure égale à celle du fft de taille minimale
        os.remove('./extrait.wav')
    fft_tab_norm=normalize_size(fft_tab, min_len_fft)
    print("min len fft=",min_len_fft)
    return fft_tab_norm



def normalize_size(T,mini):
    normalized_tab=[]
    for tab in T:
        normalized_tab.append(tab[:mini])
        
    return normalized_tab

def distance_eucl(tab1,tab2):
    temp=[]
    for i in range(0,len(tab1)-1):
        e=(tab1[i]-tab2[i])**2
        temp.append(e)
    s=0
    for e in temp:
        s=s+e
    s=np.abs(s)
    if s==0.0:
        return 1.0
    s=np.log(s)
    return s
        
#Problème de taille de tableau

def compare (seuil,tab_fft):#vérifier les ordres de grandeur entre le seuil et la distance
    similaires=[]
    print("len fft",len(tab_fft))
    for i in range(0,len(tab_fft)-2):
        for j in range(i+1,len(tab_fft)-1):
            #print(distance_eucl(tab_fft[i],tab_fft[j]))
            if distance_eucl(tab_fft[i],tab_fft[j])<=seuil:
                tuple1=(i,j)
                similaires.append(tuple1)
    print(similaires)
    return similaires


file_path="./Get_lucky.wav"

test = get_all_fft(file_path,2)
compare(37,test)

    