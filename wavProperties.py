import librosa
import soundfile
import numpy as np
import os

class WavProperties:
    def __init__(self):
        self.filename = None
        self.actor = None
        self.emotion = None
        self.featureForTraining = None
        self.stft = None
        #Mel Frequency Cepstral Coefficients
        self.mfccs = None
        #Chroma
        self.chroma = None
        self.mel = None
        #Spectogram of frequency
        self.xdb = None
        #root mean square
        self.rms = None
        #Zero crossing rate
        self.zrcc = None
        #Tempogram
        self.tempogram = None

    def extract_file_informaton(self, file_name,emotions):
        self.filename = file_name
        split_file_name = file_name.split("-")
        self.actor = split_file_name[-1].split('.')[0]
        self.emotion = emotions[split_file_name[2]]

    # Extract features (mfcc, chroma, mel) from a sound file
    def extract_feature(self,filename):
        with soundfile.SoundFile(filename) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            temp_stft = np.abs(librosa.stft(X))
            self.stft = temp_stft.tolist()
            result = np.array([])
            temp_mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            self.mfccs = temp_mfccs.tolist()
            result = np.hstack((result, temp_mfccs))
            temp_chroma = np.mean(librosa.feature.chroma_stft(S=self.stft, sr=sample_rate).T, axis=0)
            self.chroma = temp_chroma.tolist()
            result = np.hstack((result, temp_chroma))
            temp_mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            self.mel = temp_mel.tolist()
            result = np.hstack((result, temp_mel))
            self.featureForTraining = result.tolist()



