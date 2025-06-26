#Setup libraries
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

#define your Frequency bands only 10 bands for now
EQ_BANDS_HZ = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
#Look to use pydub to load audio files as librosa does not support mp3

#Load Speaker profile
#def load_speaker_profile(file_path):
#    with open(file_path, 'r') as f:
#        return json.load(f)
    # Check where to find a json/csv file for the speaker profile

#loading a csv file of the speaker profile
def load_band_eq_from_csv(csv_path, band_freqs = EQ_BANDS_HZ, coloumn = 'equalization'):
    df = pd.read_csv(csv_path)
    profile = []
    for band in band_freqs:
        #find the row with the frequency closest to the band
        idx = (np.abs(df['frequency']-band)).argmin()
        value = df.loc[idx, coloumn]
        profile.append(value)
    print(len(profile))
    return np.array(profile)    

#analyse song using librosa should use power of spectoram squared spectogram) set window to 2048, 
def compute_band_energy(y, sr, bands = EQ_BANDS_HZ):
    S = np.abs(librosa.stft(y, n_fft=2048))**2
    #count frequencies in file
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band_energies = []
    # now need to loop through Defined bands and keep a width of 1 octave (3dB range in power)
    for i in range(len(bands)):
        low = bands[i] / np.sqrt(2)
        high = bands[i] * np.sqrt(2)
        idx = np.where((freqs >= low) & (freqs <= high)) [0]
        band_energy = np.mean(S[idx]) if idx.size > 0 else 0
        #save energy profile as dB scale
        band_energies.append(10 * np.log10(band_energy + 1e-10))
    return np.array(band_energies)
def generate_eq_curve(song_profile, speaker_profile, max_gain_db=6.0):
    song_profile = np.array(song_profile)
    speaker_profile = np.array(speaker_profile)
    target = np.mean(song_profile)
    compensation = target - (speaker_profile + song_profile)
    return np.clip(compensation, -max_gain_db, max_gain_db)
def main(song_path, speaker_profile_path):
    #load audio
    y, sr = librosa.load(song_path, sr=None, mono=True)
    
    #analyse song
    song_profile = compute_band_energy(y, sr)
    
    #Load speaker profile (assumed to be a list of 10 values)
    speaker_profile = load_band_eq_from_csv(speaker_profile_path)
    #Generate EQ
    eq_curve = generate_eq_curve(song_profile, speaker_profile)
    
    #Display 
    for freq, gain in zip(EQ_BANDS_HZ, eq_curve):
        print(f"{int(freq)} Hz: {gain:+.2f} dB")
        
    #plot
    plt.figure(figsize=(10,4))
    plt.plot(EQ_BANDS_HZ, eq_curve, marker='o', label='suggested EQ')
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('dB Gain')
    plt.title('EQ Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main("Audio\Jon Shuemaker - Eclipsed by the Sun.mp3", "Speakers/Samsung Galaxy Buds.csv")
