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
from scipy.fft import fft, fftfreq

    
file_path="./Get_lucky.wav"
samples, sampling_rate = librosa.load(file_path,sr=None, mono=True, offset=0.0,duration=None )        


#fft(file_path)        
        
fs, data = wavfile.read(file_path)
data = np.mean(data, axis=1)

n=len(data)
T=1/sampling_rate

N = data.shape[0]
L = N / fs
f, ax = plt.subplots()
#ax.plot(np.arange(N) / fs, data)
ax.set_xlabel('Frequence')
ax.set_ylabel('Amplitude')
y=scipy.fft(data)
x=np.linspace(0,1//(2*T),n)
plt.plot(x,y)
plt.show



      

"""tempo, beats = librosa.beat.beat_track(y=samples, sr=sampling_rate)
print(librosa.frames_to_time(beats, sr=sampling_rate))

t1=0.25
t2=0.77
t1 = t1 * 1000 #Works in milliseconds
t2 = t2 * 1000
newAudio = AudioSegment.from_wav(file_path)
newAudio = newAudio[t1:t2]
newAudio.export('newSong.wav', format="wav") #Exports to a wav file in the current path.
    

    

plt.figure()
librosa.display.waveplot(y=samples, sr=sampling_rate)
plt.show()"""