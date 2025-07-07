import tkinter as tk
from tkinter import filedialog
import numpy as np
import sounddevice as sd
import scipy.signal as signal
import librosa

#Create Tkinter loop
root = tk.Tk()

#Definitions
EQ_BANDS = [32.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
sample_rate = 44100
block_size = 1024

#GUI-based gain array (initially 0 dB for all bands)
eq_gains_db = [tk.DoubleVar(value=0.0) for _ in range(10)]
#Global for audio data and playback position
audio_data = None
poition = 0
#Filter design (called on every block to get updated filters)
def create_filters():
    filters = []
    for i, freq in enumerate(EQ_BANDS):
        gain_db = eq_gains_db[i].get()
        band =[freq /np.sqrt(2), freq * np.sqrt(2)]
        sos = signal.iirfilter(2,[band[0]/ (sample_rate/2), band[1]/(sample_rate/2)], btype = 'band', ftype = 'butter', output = 'sos')
        filters.append((sos, gain_db))
    return filters

def apply_eq(data, filters):
    output = np.zeros_like(data)
    for sos, gain_db in filters:
        filtered = signal.sosfilt(sos, data)
        gain = 10 ** (gain_db / 20)
        output += filtered * gain
    return output

def callback (outdata, frames, time, status):
    global position
    if position + frames >= len (audio_data):
        outdata[:] = np.zeros((frames,1))
        raise sd.CallbackStop()
    chunk = audio_data[position:position+frames]
    filters = create_filters()
    eq_chunk = apply_eq(chunk,filters)
    outdata[:, 0] = eq_chunk
    position += frames
    
def play_audio():
    global position
    position = 0
    with sd.OutputStream(callback = callback, samplerate = sample_rate, channels = 1, blocksize = block_size):
         sd.sleep(int(len(audio_data)/sample_rate *1000))
def load_audio():
    global audio_data
    file_path = filedialog.askopenfilename(filetypes = [("MP3 files", "*.mp3")])
    y, _= librosa.load(file_path, sr= sample_rate, mono = True)
    audio_data = y
#Build EQ GUI
root = tk.Tk()
root.title("10-band EQ")
frame = tk.Frame(root)
frame.pack()

for i, freq in enumerate(EQ_BANDS):
    col = tk.Frame(frame)
    col.pack(side = tk.LEFT, padx = 5)
    slider = tk.Scale(col, variable = eq_gains_db[i], from_=12, to =-12, resolution = 0.5, label = f"{int(freq)}Hz", length = 200)
    slider.pack()

btn_load = tk.Button(root, text ="Load MP3", command = load_audio)
btn_load.pack(pady =5)

btn_play = tk.Button(root, text= "Play", command = play_audio)
btn_play.pack(pady = 5)

root.mainloop()
