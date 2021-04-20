
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
import random
import soundfile



def get_fft(data):
   yf=rfft(data)
   return yf

def audiosegment_to_librosawav(audiosegment):    
    samples = audiosegment.get_array_of_samples()
    arr = np.array(samples).astype(np.float32)/32768 # 16 bit 
    arr = librosa.core.resample(arr, audiosegment.frame_rate*2, 22050, res_type='kaiser_best') 
    return arr

#Affiche la transformée de fourier
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
        data = newAudio.get_array_of_samples()
        fft=get_fft(data)
        if i==1:    #On cherche le fft de longueur minimale
            min_len_fft=len(fft)
        else:
            if min_len_fft>len(fft):
                min_len_fft=len(fft)
        fft_tab.append(fft)     #On retire à tous les fft leurs derniers éléments afin qu'ils aient tous la même longueure égale à celle du fft de taille minimale
    fft_tab_norm=normalize_size(fft_tab, min_len_fft)
    print("Taille du plus cours fft =",min_len_fft)
    return fft_tab_norm



def normalize_size(T,mini):
    normalized_tab=[]
    for tab in T:
        normalized_tab.append(tab[:mini])
    return normalized_tab

def distance_eucl(tab1,tab2):  #A optimiser
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
            s+=np.nan()
        else:     
            s+=(tab1[i]*np.log(tab1[i]/tab2[i])-tab1[i]+tab2[i])
    s=np.abs(s)
    return s


def distanceIS(tab1,tab2):
    s=0
    for i in range (0,len(tab1)-1):
        if tab2[i]==0:
            s+=np.nan()
        else:
            s+=(tab1[i]/tab2[i]-np.log(tab1[i]/tab2[i])-1)
    s=np.abs(s)
    return s


def make_sym(a):
    w, h = a.shape
    a[w - w // 2 :, :] = np.flipud(a[:w // 2, :])
    a[:, h - h // 2:] = np.fliplr(a[:, :h // 2])                


def matrice_dist (tab_fft,type_distance):
    matrice=np.ones((len(tab_fft),len(tab_fft)))
    matrice=matrice*(-1)
    print("nombre de fft ",len(tab_fft))
    for i in range(0,len(tab_fft)-2):
        for j in range(i+1,len(tab_fft)-1):
            if type_distance==1:
                matrice[i][j]=distance_eucl(tab_fft[i],tab_fft[j])
            if type_distance==2:
                matrice[i][j]=distanceKL(tab_fft[i],tab_fft[j])
            if type_distance==3:
                matrice[i][j]=distanceIS(tab_fft[i],tab_fft[j])
    return matrice
def dist_3_by_3(tab_fft,type_dist):
    matrice=np.ones((len(tab_fft),len(tab_fft)))
    matrice=matrice*(-1)
    for i in range(0,len(tab_fft)-1):
        if i==0:                #On gère les cas où les intervalles en questions sont sur les extrémités et donc ne sont pas entouré par d'autres
            vector1=np.concatenate((tab_fft[i],tab_fft[i+1],tab_fft[i+2]))      
        if i==len(tab_fft)-1:
            vector1=np.concatenate((tab_fft[i-2],tab_fft[i-1],tab_fft[i]))
        else:
            vector1=np.concatenate((tab_fft[i-1],tab_fft[i],tab_fft[i+1]))
        for j in range(i,len(tab_fft)-1):
            if j==0:               
                vector2=np.concatenate((tab_fft[j],tab_fft[j+1],tab_fft[j+2]))      
            if j==len(tab_fft)-1:
                vector2=np.concatenate((tab_fft[j-2],tab_fft[j-1],tab_fft[j]))
            else:
                vector2=np.concatenate((tab_fft[j-1],tab_fft[j],tab_fft[j+1]))
                
            if type_distance=='eucl':
                matrice[i][j]=distance_eucl(vector1,vector2)
            if type_distance=='IS':
                matrice[i][j]=distanceKL(vector1,vector2)
            if type_distance=='KL':
                matrice[i][j]=distanceIS(vector1,vector2)
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
            if liste[k]<m[0] and k!=i and liste[k]>0.001 and liste[k]!=-1:
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

def proba(prob):
    rand=random.random()
    if rand<prob:
        return True
    else:
        return False
    

def generate(file_path,newtimecodes,length,prob):
    orig, sr = librosa.load(file_path)
    if len(newtimecodes)==0:
        print('Aucune similarité détecté avec ces paramètres')
        return False
    tab_duree=[]
    duree=0
    newAudio = AudioSegment.from_wav(file_path)
    finalArray=[]
    debut=newtimecodes[0][1]
    duree+=newtimecodes[0][0]
    tab_duree.append(int(duree))
    while duree<length:
        
        fin=newtimecodes[0][0]
        j=0
        while fin<debut+0.001 or proba(prob)==False:
            if j==len(newtimecodes)-1:
                j=0
            j+=1
            fin=newtimecodes[j][0]
        intervalle = newAudio[debut*1000:fin*1000]
        array = audiosegment_to_librosawav(intervalle)  #On transforme l'audio en un tableau exploitable par librosa
        out = array.copy()                              #La copie permet de ne pas modifier le fichier originel
        apply_fadein(out, sr, 0.1)
        apply_fadeout(out, sr, 0.1)
        finalArray=np.concatenate((finalArray,out))     
        duree+=abs(fin-debut)
        tab_duree.append(int(duree))
        debut=newtimecodes[j][1]
    soundfile.write('InfiniteAudio.wav', finalArray, samplerate=sr)
    return tab_duree


def apply_fadeout(audio, sr, duration):
    length = int(duration*sr)
    end = audio.shape[0]
    start = end - length
    fade_curve = x = np.linspace(-np.pi/2,0,length)
    fade_curve = -np.sin(x)
    audio[start:end] = audio[start:end] * fade_curve
    
def apply_fadein(audio, sr, duration):
    length = int(duration*sr)
    end = length
    start = 0
    fade_curve = np.linspace(0.0, np.pi/2, length)
    fade_curve = np.sin(fade_curve)
    audio[start:end] = audio[start:end] * fade_curve    
    
file_path="./Get_lucky.wav"

tps1=time.time()
test = get_all_fft(file_path,1)
tps2=time.time()

print("temps pour récupérer les ffts :",tps2-tps1," secondes")



mat=matrice_dist(test,'eucl')
np.save('matrice_dist_10.txt', mat)
tps3=time.time()

mm=np.load('matrice_dist.npy')
mm=make_sym(mm)

print("temps pour comparer tous les ffts :",tps3-tps2," secondes")


print(np.shape(mm))
tc=np.load('timecodes.npy')

display(mm)
t=tri(33,mm)
ntc=inter_to_tc(t,tc)
print(generate(file_path,ntc,300,1))
tps4=time.time()
print("temps pour générer le morceau de 5min :",tps4-tps3," secondes")
