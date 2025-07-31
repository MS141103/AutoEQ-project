'''MIT License

Copyright (c) 2021 Ramadan Ibrahem

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''
import Song_analyser
import wave
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QWidget, QSlider, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QMainWindow)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import threading
from Song_analyser import generate_eq_curve, compute_band_energy, load_band_eq_from_csv  # Ensure this function returns a 10-band EQ curve

EQ_BANDS = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
print(sd.query_devices())
# Apply EQ with simple gain adjustments (placeholder logic)
def apply_eq(audio_data, eq_gains_db):
    gains_linear = 10 ** (np.array(eq_gains_db) / 20)
    return audio_data * np.mean(gains_linear)

class AudioPlaybackThread(QThread):
    finished = pyqtSignal()

    def __init__(self, audio_data, samplerate):
        super().__init__()
        self.audio_data = audio_data
        self.samplerate = samplerate
        self._running = True
        self.paused = False

    def run(self):
        try:
            with sd.OutputStream( samplerate=self.samplerate, channels=1) as stream:
                i = 0
                block_size = 1024
                while self._running and i < len(self.audio_data):
                    if not self.paused:
                        block = self.audio_data[i:i+block_size]
                        if len(block) < block_size:
                            block = np.pad(block, (0, block_size - len(block)))
                        stream.write(block.astype(np.float32))
                        i += block_size
                    else:
                        self.msleep(100)
        except Exception as e:
            print("Audio playback error:", e)
        self.finished.emit()

    def stop(self):
        self._running = False

    def toggle_pause(self):
        self.paused = not self.paused

class EqualizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("10-Band EQ Player")
        self.layout = QVBoxLayout()

        # EQ Sliders
        self.eq_sliders = []
        slider_layout = QHBoxLayout()
        for i in range(10):
            slider = QSlider(Qt.Vertical)
            slider.setMinimum(-10)
            slider.setMaximum(10)
            slider.setValue(0)
            slider_layout.addWidget(slider)
            self.eq_sliders.append(slider)
        self.layout.addLayout(slider_layout)

        # Buttons
        self.load_btn = QPushButton("Load WAV File")
        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")

        self.layout.addWidget(self.load_btn)
        self.layout.addWidget(self.play_btn)
        self.layout.addWidget(self.pause_btn)
        self.layout.addWidget(self.stop_btn)

        self.setLayout(self.layout)

        # Connect buttons
        self.load_btn.clicked.connect(self.load_audio_file)
        self.play_btn.clicked.connect(self.start_playback)
        self.pause_btn.clicked.connect(self.pause_playback)
        self.stop_btn.clicked.connect(self.stop_playback)

        self.audio_data = None
        self.samplerate = 44100
        self.audio_thread = None
        
        #graph setup
        self.plot_canvas = FigureCanvas(Figure(figsize = (6, 3)))
        self.layout.addWidget(self.plot_canvas)
        
    def plot_eq_curve(self, eq_curve):
        bands = [f"Band {i+1}" for i in range(len(eq_curve))]
        plt.figure("Generated EQ Curve")
        plt.bar(bands, eq_curve)
        plt.xlabel("Frequency Bands")
        plt.ylabel("Gain (dB)")
        plt.title("Auto-Generated EQ Curve")
        plt.tight_layout()
        plt.show()

    def plot_frequency_decomposition(self, song_profile):
        bands = [f"Band {i+1}" for i in range(len(song_profile))]
        plt.figure("Song Frequency Decomposition")
        plt.bar(bands, song_profile)
        plt.xlabel("Frequency Bands")
        plt.ylabel("Relative Energy")
        plt.title("Frequency Distribution of Loaded Song")
        plt.tight_layout()
        plt.show()
    def show_eq_plot(self, eq_curve, song_profile):
        self.plot_canvas.figure.clf()
        ax = self.plot_canvas.figure.add_subplot(211)
        ax.plot([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000], eq_curve, marker='o')
        ax.set_xscale('log')
        ax.set_title('EQ Curve')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain (dB)')
        ax.grid(True)

        ax2 = self.plot_canvas.figure.add_subplot(212)
        ax2.bar([str(int(f)) for f in [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]], song_profile)
        ax2.set_title('Song Frequency Decomposition')
        ax2.set_ylabel('dB Energy')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.grid(True)

        self.plot_canvas.draw()

    def load_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open WAV File", "", "WAV files (*.wav)")
        if file_path:
            try:
                with wave.open(file_path, 'rb') as wav:
                    self.samplerate = wav.getframerate()
                    n_frames = wav.getnframes()
                    audio_data = np.frombuffer(wav.readframes(n_frames), dtype=np.int16)
                    self.audio_data = audio_data.astype(np.float32) / 32768.0

                # Prompt for speaker profile
                speaker_path, _ = QFileDialog.getOpenFileName(self, "Load Speaker Profile", "", "CSV files (*.csv)")
                if speaker_path:
                    speaker_profile = load_band_eq_from_csv(speaker_path)
                    song_profile = compute_band_energy(self.audio_data, self.samplerate)
                    eq_curve = generate_eq_curve(song_profile, speaker_profile)

                    # Update sliders
                    for i, val in enumerate(eq_curve):
                        if i < len(self.eq_sliders):
                            slider_val = int(round(val))
                            self.eq_sliders[i].setValue(max(-10, min(10, slider_val)))
            except Exception as e:
                print("Error loading WAV file:", e)
        self.show_eq_plot(eq_curve, song_profile)
    def start_playback(self):
        if self.audio_data is None:
            return

        eq_gains_db = [slider.value() for slider in self.eq_sliders]
        processed_audio = apply_eq(self.audio_data, eq_gains_db)

        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.wait()

        self.audio_thread = AudioPlaybackThread(processed_audio, self.samplerate)
        self.audio_thread.start()

    def pause_playback(self):
        if self.audio_thread:
            self.audio_thread.toggle_pause()

    def stop_playback(self):
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.wait()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EqualizerApp()
    window.show()
    sys.exit(app.exec_())


