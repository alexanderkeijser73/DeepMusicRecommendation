from __future__ import print_function
import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def get_mfcc(audio_signal,sr):
    """
    Calculates mfcc, first and second order difference features from raw audio
    :param audio_signal: audio as time series in 1D float array
    :param sr:  sample rate for audio file
    :return:    concatenated mfcc and difference features
    """
    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sr, hop_length=hop_length, n_mfcc=13, n_fft=1024)

    # And the first- and second-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2)).T

    return features

def learn_dictionary(audio_files, k=4000):
    """
    Learns BOW dictionary of k elements using k means
    :param audio_files: array with rows containing songs as audio time series
    :param k:           vocabulary size (number of clusters)
    """
    features = np.empty((0, 39), dtype=np.float)
    for y in audio_files:
        mfccs = get_mfcc(y, sr=22050)
        features = np.append(features, mfccs, axis=0)
    print("Training k-means clustering...")
    means = KMeans(n_clusters=k).fit(features)
    return means

def vector_quantize(song, kmeans):
    """
    Generates BOW representation by counting cluster assignments for song MFCCs
    :param song:    1D array containing time series
    :param kmeans:  trained k-Means object
    :return:        counts of mean selections
    """
    mfcc = get_mfcc(song, sr=22050)
    # Find cluster assignment counts
    assignments = kmeans.predict(mfcc)

    counts = np.zeros(kmeans.cluster_centers_.shape[0])
    for i in assignments:
        counts[i] += 1

    return counts

def generate_data_matrix(songs, k=200):
    """
    Generates data matrix for array of song time series and applies PCA
    :param songs: array with rows containing songs as audio time series
    :return:      data matrix after applying PCA to retain 95 percent variance
    """
    # Train k-means clustering
    kmeans = learn_dictionary(songs, k=k)

    # Calculate BOW features for songs
    data_matrix = np.empty((0, k), dtype=np.float)
    for y in songs:
        counts = vector_quantize(y, kmeans)
        data_matrix = np.append(data_matrix, np.expand_dims(counts, axis=0), axis=0)

    # Apply PCA to data matrix
    pca = PCA(n_components=0.95, svd_solver='full')
    data_matrix_reduced = pca.fit_transform(data_matrix)

    return data_matrix_reduced


if __name__ == '__main__':
    # 1. Get the file path to the included audio example
    filename = librosa.util.example_audio_file()

    # 2. Load the audio as a waveform `y`
    #    Store the sampling rate as `sr`
    y, sr = librosa.load(filename)

    # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
    hop_length = 512

    # Calculate BOW representation
    print(get_mfcc(y, sr).shape)
    kmeans = learn_dictionary([y], k=100)
    print(vector_quantize(y, kmeans))
    print(generate_data_matrix([y, y+0.1]))
