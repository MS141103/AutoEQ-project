#import libraries
import sounddevice as sd
import numpy as np
import scipy.signal as signal
import librosa

#define EQ gain per band 
eq_gains_db = [-2, -6, -6, 6, 6, 4, 2, 1, 1, 1]
EQ_BANDS_HZ = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
sample_rate = 44100

#designing bandpass filters for each band
def create_band_filters(eq_gains_db, sr = 44100):
    filters = []
    for gain_db, center_freq in zip (eq_gains_db, EQ_BANDS_HZ):
        #designing band filter will keep 1 octave width again
        band = [center_freq / np.sqrt(2), center_freq * np.sqrt(2)]
        sos = signal.iirfilter(2, [band[0]/(sr), band[1]/(sr)], btype = 'band', ftype = 'butter', output= 'sos')
        filters.append((sos, gain_db))
    return filters


#apply filters with generated eq gain from Song analyser
def apply_eq(data, filters):
    eq_out = np.zeros_like(data)
    for sos, gain_db in filters:
        filtered = signal.sosfilt(sos, data)
        gain = 10 ** (gain_db / 20)
        eq_out += filtered * gain
    return eq_out
#loading audio file
filename = r"C:\Users\saiffmua\Documents\Contour_Project\AutoEQ-project\Audio\Jon Shuemaker - Eclipsed by the Sun.mp3"
audio, sr = librosa.load(filename, sr=sample_rate, mono = True)

filters = create_band_filters(eq_gains_db, sr)

#defining a playback callback
block_size = 1024
position = 0

def callback (outdata, frames, time, status):
    global position
    chunk = audio[position:position+frames]
    if len(chunk) < frames:
        outdata[:len(chunk), 0] = apply_eq(chunk, filters)
        outdata[len(chunk):] = 0
        raise sd.Callbackstop()
    else:
        outdata[:, 0] = apply_eq(chunk, filters)
    position += frames

#start streaming
with sd.OutputStream(callback = callback, samplerate = sample_rate, channels = 1, blocksize = block_size):
    print("playing with real_time EQ...")
    sd.sleep(int((len(audio) / sample_rate)*1000))