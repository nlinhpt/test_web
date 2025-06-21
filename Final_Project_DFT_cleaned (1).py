import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import io
from scipy.io.wavfile import write as wav_write
from scipy.signal import butter, filtfilt

# Function: Analyze Spectrum
def analyze_frequency(signal, sr):
    n = len(signal)
    signal_fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=1/sr)
    pos_freqs = freqs[:n // 2]
    pos_magnitude = np.abs(signal_fft)[:n // 2]

    plt.figure(figsize=(10, 6))
    plt.plot(pos_freqs, pos_magnitude, color='blue')
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    st.pyplot(plt.gcf())

    return pos_freqs, pos_magnitude

# Function: Filter Audio
def filter_audio(signal, sr, cutoff_freq, filter_type='low-pass'):
    nyquist = sr / 2
    if filter_type == 'low-pass':
        b, a = butter(N=2, Wn=cutoff_freq / nyquist, btype='low')
    elif filter_type == 'high-pass':
        b, a = butter(N=2, Wn=cutoff_freq / nyquist, btype='high')
    else:
        raise ValueError("Invalid filter_type. Use 'low-pass' or 'high-pass'.")
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

st.title("Audio Signal Processing with Streamlit")
st.write("Upload an audio file (MP3, WAV, FLAC) to analyze its spectrum and filter the signal.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])
if uploaded_file is not None:
    st.write("Processing Uploaded File...")

    # Đọc file với librosa
    audio_data, sr = librosa.load(uploaded_file, sr=None, mono=True)
    signal = audio_data

    # Hiển thị audio gốc
    buf = io.BytesIO()
    wav_write(buf, sr, (signal * 32767).astype(np.int16))
    st.audio(buf, format="audio/wav")

    st.write(f"Sample Rate: {sr} Hz")
    st.write(f"Number of Samples: {len(signal)}")

    st.write("Frequency Spectrum of the Original Signal:")
    analyze_frequency(signal, sr)

    st.write("Apply a Low-Pass Filter:")
    cutoff_freq = st.slider("Cutoff Frequency (Hz)", min_value=100, max_value=sr // 2, value=1000)
    filtered_signal = filter_audio(signal, sr, cutoff_freq, filter_type='low-pass')

    # Phát lại âm thanh đã lọc
    buf_filtered = io.BytesIO()
    wav_write(buf_filtered, sr, (filtered_signal * 32767).astype(np.int16))
    st.audio(buf_filtered, format="audio/wav")

    st.write("Frequency Spectrum of the Filtered Signal:")
    analyze_frequency(filtered_signal, sr)
