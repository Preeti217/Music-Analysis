import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from wavProperties import WavProperties

# Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


# get audio summary for a .wav file
# def get_audio_summary(filename):
#     # 'filename' must be absolute or working directory should be pointed to the folder containing 'filename'
#     if filename.endsWith('.wav'):
#         audio_segment = AudioSegment.from_file(filename)
#         print(f"Channels: {audio_segment.channels}")
#         print(f"Sample width: {audio_segment.sample_width}")
#         print(f"Frame rate (sample rate): {audio_segment.frame_rate}")
#         print(f"Frame width: {audio_segment.frame_width}")
#         print(f"Length (ms): {len(audio_segment)}")
#         print(f"Frame count: {audio_segment.frame_count()}")
#         print(f"Intensity: {audio_segment.dBFS}")
#
#     print(f"End")
#     return


# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, wav_file, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
            wav_file.stft = stft.tolist()
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            wav_file.mfccs = mfccs.tolist()
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            wav_file.chroma = chroma.tolist()
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            wav_file.mel = mel.tolist()
            result = np.hstack((result, mel))
        wav_file.featureForTraining = result.tolist()
    return


# data visualization
# def compare_intra_actor_emotions(emotions_dict):
#     #pick randomly one file from every emotion
#
#     for emotion in emotions_dict


# {actor : {emotion: [files]}}
def preprocess_data():
    actor_emotions_dict = {}
    temp = glob.glob("/Users/preetinarayanan/Desktop/speech-emotion-recognition-ravdess-data/Actor_*/*.wav")
    for file in temp:
        file_name = os.path.basename(file)
        wav_file = WavProperties()
        wav_file.extract_file_informaton(file_name, emotions)
        extract_feature(file,wav_file,mfcc=True, chroma=True, mel=True)
        emotions_dict = {}
        if wav_file.actor in actor_emotions_dict:
            emotions_dict = actor_emotions_dict[wav_file.actor]
            if wav_file.emotion in emotions_dict:
                emotions_dict[wav_file.emotion].append(wav_file)
            else:
                emotions_dict[wav_file.emotion] = [wav_file]
        else:
            actor_emotions_dict[wav_file.actor] = {wav_file.emotion: [wav_file]}
    return actor_emotions_dict


# Emotions to observe
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']


# Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x, y = [], []
    temp = glob.glob("/Users/preetinarayanan/Desktop/speech-emotion-recognition-ravdess-data/Actor_*/*.wav")
    for file in temp:
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        wav_file = WavProperties()
        wav_file.extract_file_informaton(file_name, emotions)
        extract_feature(file, wav_file, mfcc=True, chroma=True, mel=True)
        feature = wav_file.featureForTraining
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


def runModel():
    # Split the dataset
    x_train, x_test, y_train, y_test = load_data(test_size=0.25)
    # Initialize the Multi Layer Perceptron Classifier
    model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,),
                          learning_rate='adaptive', max_iter=500)
    # Train the model
    model.fit(x_train, y_train)
    # Predict for test score
    y_pred = model.predict(x_test)
    # Calculate the accuracy of our model
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    # Print the accuracy
    print("Accuracy: {:.2f}%".format(accuracy * 100))
