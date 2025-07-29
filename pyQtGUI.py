import sys
import numpy as np
import sounddevice as sd
import librosa
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QWidget, QSlider, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog)
from PyQt5.QtCore import Qt
from song_analyser import generate_eq_curve  # Ensure this function returns a 10-band EQ curve

EQ_BANDS = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024

class Equalizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto-EQ Sound Equalizer with Visualization")
        self.eq_gains_db = [0.0] * 10
        self.auto_eq_curve = None
        self.audio_data = None
        self.position = 0
        self.playing = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        sliders_layout = QHBoxLayout()
        self.sliders = []

        for freq in EQ_BANDS:
            slider_layout = QVBoxLayout()
            label = QLabel(f"{int(freq)} Hz")
            slider = QSlider(Qt.Vertical)
            slider.setMinimum(-120)
            slider.setMaximum(120)
            slider.setValue(0)
            slider.setTickInterval(10)
            slider.setTickPosition(QSlider.TicksRight)
            slider.valueChanged.connect(self.update_gains)
            self.sliders.append(slider)
            slider_layout.addWidget(label)
            slider_layout.addWidget(slider)
            sliders_layout.addLayout(slider_layout)

        self.load_button = QPushButton("Load WAV")
        self.load_button.clicked.connect(self.load_audio)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_audio)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_audio)

        self.auto_eq_button = QPushButton("Auto EQ")
        self.auto_eq_button.clicked.connect(self.apply_auto_eq)

        # Matplotlib figures for EQ curve and spectrogram
        self.figure, (self.ax_eq, self.ax_spec) = plt.subplots(2, 1, figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)

        layout.addLayout(sliders_layout)
        layout.addWidget(self.load_button)
        layout.addWidget(self.play_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.auto_eq_button)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_gains(self):
        self.eq_gains_db = [slider.value() / 10.0 for slider in self.sliders]
        self.update_visuals()

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open WAV file", "", "WAV files (*.wav)")
        if file_path:
            y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            self.audio_data = y
            self.update_visuals()

    def play_audio(self):
        if self.audio_data is None:
            return
        self.playing = True
        self.position = 0
        with sd.OutputStream(callback=self.callback, samplerate=SAMPLE_RATE,
                              channels=1, blocksize=BLOCK_SIZE):
            while self.playing and self.position < len(self.audio_data):
                sd.sleep(100)

    def pause_audio(self):
        self.playing = False

    def callback(self, outdata, frames, time, status):
        if not self.playing or self.position + frames >= len(self.audio_data):
            outdata[:frames, 0] = 0
            raise sd.CallbackStop()

        chunk = self.audio_data[self.position:self.position+frames]
        self.update_gains()
        filters = self.create_band_filters()
        eq_chunk = self.apply_eq(chunk, filters)
        outdata[:, 0] = eq_chunk
        self.position += frames

    def create_band_filters(self):
        filters = []
        for gain_db, freq in zip(self.eq_gains_db, EQ_BANDS):
            band = [freq / np.sqrt(2), freq * np.sqrt(2)]
            sos = signal.iirfilter(2, [band[0]/(SAMPLE_RATE/2), band[1]/(SAMPLE_RATE/2)],
                                   btype='band', ftype='butter', output='sos')
            filters.append((sos, gain_db))
        return filters

    def apply_eq(self, data, filters):
        eq_out = np.zeros_like(data)
        for sos, gain_db in filters:
            filtered = signal.sosfilt(sos, data)
            gain = 10 ** (gain_db / 20)
            eq_out += filtered * gain
        return eq_out

    def apply_auto_eq(self):
        if self.audio_data is None:
            return
        self.auto_eq_curve = generate_eq_curve(self.audio_data, SAMPLE_RATE)
        for i, gain in enumerate(self.auto_eq_curve):
            self.sliders[i].setValue(int(gain * 10))
        self.update_gains()

    def update_visuals(self):
        self.ax_eq.clear()
        self.ax_spec.clear()

        # Plot current EQ curve
        self.ax_eq.semilogx(EQ_BANDS, self.eq_gains_db, marker='o', label='Manual EQ')
        if self.auto_eq_curve is not None:
            self.ax_eq.semilogx(EQ_BANDS, self.auto_eq_curve, marker='x', linestyle='--', color='red', label='Auto EQ')
        self.ax_eq.set_title("EQ Curve")
        self.ax_eq.set_xlabel("Frequency (Hz)")
        self.ax_eq.set_ylabel("Gain (dB)")
        self.ax_eq.legend()
        self.ax_eq.grid(True, which='both', ls='--')

        # Plot spectrogram of loaded audio
        if self.audio_data is not None:
            S = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data)), ref=np.max)
            self.ax_spec.imshow(S, aspect='auto', origin='lower', cmap='magma')
            self.ax_spec.set_title("Spectrogram")
            self.ax_spec.set_ylabel("Frequency bins")

        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    eq = Equalizer()
    eq.show()
    sys.exit(app.exec_())

