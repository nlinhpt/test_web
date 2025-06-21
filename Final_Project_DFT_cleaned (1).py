import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import soundfile as sf
from scipy.io.wavfile import write
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

# Streamlit App
st.title("Audio Signal Processing with Streamlit")
st.write("Upload an audio file to analyze its spectrum and filter the signal.")

# Upload file section
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])
if uploaded_file:
    # Load audio file
    st.write("Processing Uploaded File...")
    audio = AudioSegment.from_file(uploaded_file)
    if audio.channels > 1:
        audio = audio.set_channels(1)
    signal = np.array(audio.get_array_of_samples(), dtype=np.float64)
    sr = audio.frame_rate

    # Display original audio
    st.audio(uploaded_file, format="audio/wav")
    st.write(f"Sample Rate: {sr} Hz")
    st.write(f"Number of Samples: {len(signal)}")

    # Analyze frequency spectrum
    st.write("Frequency Spectrum of the Original Signal:")
    analyze_frequency(signal, sr)

    # Filter audio
    st.write("Apply a Low-Pass Filter:")
    cutoff_freq = st.slider("Cutoff Frequency (Hz)", min_value=100, max_value=sr // 2, value=1000)
    filtered_signal = filter_audio(signal, sr, cutoff_freq, filter_type='low-pass')

    # Display filtered audio
    filtered_audio = AudioSegment(
        filtered_signal.astype(np.int16).tobytes(),
        frame_rate=sr,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    st.audio(filtered_audio.export(format="wav"), format="audio/wav")

    # Analyze filtered frequency spectrum
    st.write("Frequency Spectrum of the Filtered Signal:")
    analyze_frequency(filtered_signal, sr)
