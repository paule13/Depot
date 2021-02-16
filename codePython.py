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

def get_fft(file_path):
   samples, sampling_rate = librosa.load(file_path,sr=None, mono=True, offset=0.0,duration=None )        
   duration = librosa.get_duration(y=samples, sr=sampling_rate)
   fs, data = wavfile.read(file_path)
   data = np.mean(data, axis=1)
   yf=rfft(data)
   print(len(yf))
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
    for i in range (1,len(time_codes)):
        newAudio = AudioSegment.from_wav(file_path)
        newAudio = newAudio[time_codes[i-1]*1000:time_codes[i]*1000]
        newAudio.export('extrait.wav', format="wav")
        fft_tab.append(get_fft('./extrait.wav'))
        os.remove('./extrait.wav')
    return fft_tab
    


file_path="./Get_lucky.wav"
get_all_fft(file_path,2)

