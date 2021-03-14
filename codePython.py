
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
import time
import h5py

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
   np.save('timecodes',new_tab) 
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
    print("Taille du plus cours fft =",min_len_fft)
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

def distanceKL(tab1,tab2):
    s=0
    for i in range (0,len(tab1)-1):
        if tab2[i]==0:
            s+=0.0001
        else:     
            s+=(tab1[i]*np.log(tab1[i]/tab2[i])-tab1[i]+tab2[i])
    s=np.abs(s)
    return s


def distanceIS(tab1,tab2):
    s=0
    for i in range (0,len(tab1)-1):
        if tab2[i]==0:
            s+=0.0001
        else:
            s+=(tab1[i]/tab2[i]-np.log(tab1[i]/tab2[i])-1)
    s=np.abs(s)
    return s


def make_sym(a):
    w, h = a.shape
    a[w - w // 2 :, :] = np.flipud(a[:w // 2, :])
    a[:, h - h // 2:] = np.fliplr(a[:, :h // 2])                


def matrice_dist (tab_fft,type_distance):
    matrice=np.zeros((len(tab_fft),len(tab_fft)))
    print("nombre de fft ",len(tab_fft))
    for i in range(0,len(tab_fft)-1):
        for j in range(0,len(tab_fft)-1):
            if type_distance==1:
                matrice[i][j]=distance_eucl(tab_fft[i],tab_fft[j])
            if type_distance==2:
                matrice[i][j]=distanceKL(tab_fft[i],tab_fft[j])
            if type_distance==3:
                matrice[i][j]=distanceIS(tab_fft[i],tab_fft[j])
    return matrice





def minimum(matrice):
    mini=[]
    for i in range(0,len(matrice)):
        liste=matrice[i]
        if i==0:
            m=(liste[1],1)
        else:   
            m=(liste[0],0)
        for k in range (0,len(liste)):
            if liste[k]<m[0] and k!=i and liste[k]>0.001:
                m=(liste[k],k)
        mini.append(m)
        

        
    return mini

def tri(seuil,matrice):
    matrice=minimum(matrice)
    trie=[]
    for j in range (0,len(matrice)):
        if matrice[j][0]<=seuil:
            trie.append((j,matrice[j][1]))
    return trie
            
def display(matrice):
    figure = plt.figure() 
    axes = figure.add_subplot(111) 
    caxes = axes.matshow(matrice,interpolation ='nearest')
    figure.colorbar(caxes) 
    plt.show()

def inter_to_tc(inter,tc):
    newtab=[]
    for e in inter:
        a=tc[e[0]]
        b=tc[e[1]]
        ntc=(a,b)
        newtab.append(ntc)
    return newtab

def generate(timecodes,file_path,newtimecodes,length):
    duree=0
    newAudio = AudioSegment.from_wav(file_path)
    debut=0
    i=0
    while duree<=length:
        fin=newtimecodes[i][0]
        j=0
        while fin<debut:
            j+=1
            fin=newtimecodes[j][0]
        i=j
        newAudio += newAudio[debut*1000:fin*1000]
        duree+=fin-debut
        debut=newtimecodes[i][1]    
    print(duree)            
    newAudio.export('infinite_audio.wav', format="wav")
    return newAudio


file_path="./Get_lucky.wav"

'''
tps1=time.time()
test = get_all_fft(file_path,1)
tps2=time.time()
print("temps pour récupérer les ffts :",tps2-tps1," secondes")


tps3=time.time()
mat=matrice_dist(test,1)
np.save('test.txt', mat)

print("temps pour comparer tous les ffts :",tps3-tps2," secondes")

'''

tc=np.load('timecodes.npy')
m = np.load('test.txt.npy')
display(m)
t=tri(31,m)
ntc=inter_to_tc(t,tc)
print(ntc)



#forwardTC=ntc[:len(ntc)//2-1]
#backwardTC=ntc[len(ntc)//2-1:len(ntc)]
generate(tc,file_path,ntc,300)



