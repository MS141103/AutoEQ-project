import sys
import numpy as np
import sounddevice as sd
import librosa
import scipy.signal as signal
from PyQt5.QtWidgets import (QApplication, QWidget, QSlider, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog)
from PyQt5.QtCore import Qt

EQ_BANDS = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024

class Equalizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto-EQ Sound Equalizer")
        self.eq_gains_db = [0.0] * 10
        self.audio_data = None
        self.position = 0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        sliders_layout = QHBoxLayout()
        self.sliders = []

        for i, freq in enumerate(EQ_BANDS):
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

        layout.addLayout(sliders_layout)
        layout.addWidget(self.load_button)
        layout.addWidget(self.play_button)
        self.setLayout(layout)

    def update_gains(self):
        self.eq_gains_db = [slider.value() / 10.0 for slider in self.sliders]

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open WAV file", "", "WAV files (*.wav)")
        if file_path:
            y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            self.audio_data = y

    def play_audio(self):
        if self.audio_data is None:
            return
        self.position = 0
        with sd.OutputStream(callback=self.callback, samplerate=SAMPLE_RATE,
                              channels=1, blocksize=BLOCK_SIZE):
            sd.sleep(int(len(self.audio_data) / SAMPLE_RATE * 1000))

    def callback(self, outdata, frames, time, status):
        if self.position + frames >= len(self.audio_data):
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    eq = Equalizer()
    eq.show()
    sys.exit(app.exec_())
