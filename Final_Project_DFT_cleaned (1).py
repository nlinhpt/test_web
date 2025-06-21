import streamlit as st
import matplotlib.pyplot as plt
#!/usr/bin/env python
# coding: utf-8

# # 🎧 Noise Reduction with Discrete Fourier Transform

# ## 📌 Introduction
# Trong kỷ nguyên số, tín hiệu âm thanh và dữ liệu thường xuyên bị "nhiễm bẩn" bởi tiếng ồn không mong muốn, làm giảm đáng kể chất lượng và thông tin. Nhằm khôi phục sự trong trẻo của dữ liệu, đề tài này tập trung khám phá và ứng dụng Phương pháp Fourier Rời rạc (Discrete Fourier Transform - DFT). DFT không chỉ là một công cụ toán học, mà còn là "cặp kính thần kỳ" cho phép chúng ta phân tích tín hiệu từ miền thời gian sang miền tần số, nơi tiếng ồn thường bộc lộ bản chất và tần số đặc trưng của nó. Bằng cách định vị và loại bỏ các thành phần nhiễu ở miền tần số, chúng ta có thể tách biệt tiếng ồn khỏi tín hiệu mong muốn một cách hiệu quả và tinh tế. Báo cáo này sẽ trình bày chi tiết cách triển khai và đánh giá hiệu quả của DFT trong việc khử nhiễu tín hiệu, hướng tới một tương lai dữ liệu rõ ràng và chính xác hơn.
# 
# 

# ## 🧮 Mathematical Background

# 

# ## 🎯 DFT and IDFT Implementation

# ### DFT and IDFT from scratch

# **Thuật toán: Discrete Fourier Transform (DFT)**
# 
# **Input:** Dãy tín hiệu đầu vào $x \in \mathbb{R}^n$
# 
# **Output:** Dãy phổ tần số $X\in \mathbb{C}^n$
# 
# 1. Chuyển $x$ thành mảng số thực (nếu chưa)
# 2. Gán $n \leftarrow \text{độ dài của } x$
# 3. Khởi tạo ma trận Fourier $F \in \mathbb{C}^{n \times n}$
# 4. Với mỗi $j = 0$ đến $n - 1$:
#    - Với mỗi $k = 0$ đến $n - 1$:
#      - $F[j][k] = e^{-2\pi i \cdot jk / n}$
# 5. Tính tích: $X = F \cdot x$
# 6. Trả về $X$
# 

# **Thuật toán: Inverse Discrete Fourier Transform (IDFT)**
# 
# **Input:** Dãy phổ tần số đầu vào $X \in \mathbb{C}^n$
# 
# **Output:** Dãy tín hiệu khôi phục $x \in \mathbb{C}^n$
# 
# 1. Chuyển $X$ thành mảng số phức
# 2. Gán $n \leftarrow \text{độ dài của } X $
# 3. Khởi tạo ma trận Fourier ngược $F_{\text{inv}} \in \mathbb{C}^{n \times n}$
# 4. Với mỗi $j = 0$ đến $n - 1$:
#    - Với mỗi $k = 0$ đến $n - 1$:
#      - $F_{\text{inv}}[j][k] = e^{+2\pi i \cdot jk / n}$
# 5. Tính tích: $x = F_{\text{inv}} \cdot X$
# 6. Chuẩn hóa: $x \leftarrow \frac{x}{n}$
# 7. Trả về $x$
# 

# In[3]:


#get_ipython().system('pip install pydub')


# In[4]:


#get_ipython().system('pip install soundfile')


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from pydub import AudioSegment
import numpy as np
import plotly.graph_objects as go
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write
from tqdm import tqdm  # Import tqdm for progress bar
from scipy.signal import butter, filtfilt, fftconvolve


# In[6]:


def dft(x):
    """
    Discrete Fourier Transform (DFT) using Fourier Matrix.
    x: array-like, shape (n,)
    return: array of complex numbers, shape (n,)
    """
    x = np.asarray(x, dtype=float)  # chuyển về dạng float
    n = x.shape[0]  # lấy số lượng sample

    # Tạo ma trận Fourier F
    F = np.zeros((n, n), dtype=complex)
    for j in range(n):  # Duyệt qua các dòng (chỉ số thời gian)
        for k in range(n):  # Duyệt qua các cột (chỉ số tần số)
            F[j][k] = np.exp(-2j * np.pi * j * k / n)  # Tính phần tử F[j][k]

    # Tính X: F[n x n] @ x [n x 1] = X [n x 1]
    X = F @ x

    return X

def idft(X):
    """
    Inverse Discrete Fourier Transform (IDFT) from scratch.
    X: array-like, shape (n,)
    return: array of complex numbers, shape (n,)
    """
    X = np.asarray(X, dtype=complex)  # Đảm bảo dữ liệu đầu vào là số phức
    n = X.shape[0]  # lấy số lượng sample

    # Tạo ma trận Inverse Fourier F
    F_inv = np.zeros((n, n), dtype=complex)
    for j in range(n):  # Duyệt qua các dòng (chỉ số thời gian)
        for k in range(n):  # Duyệt qua các cột (chỉ số tần số)
            F_inv[j][k] = np.exp(2j * np.pi * j * k / n)  # Tính phần tử F[j][k]

    # Tính X: F_inv [n x n] @ X [n x 1] = x [n x 1]
    x = F_inv @ X

    return x / n


# ### 🔊 Audio Signal Example

# #### 🧪 Example 5.8.3

# 
# ### Bài toán
# 
# Giả sử ta đặt một microphone dưới một chiếc trực thăng đang lơ lừng, trong vòng 1 giây micro ghi lại tín hiệu âm thanh như biểu đồ hình 5.8.3. Tín hiệu có nhiều dao động, nhưng do nhiễu nên không rõ ràng.
# 
# **Mục tiêu:** Dùng DFT để phân tích tín hiệu và tìm ra những tần số chính.
# 
# 
# ### Mô hình tín hiệu và giả định
# 
# Ta giả định tín hiệu thu được có dạng:
# 
# $y(\tau) = \cos(2\pi \cdot 80 \tau) + 2 \sin(2\pi \cdot 50 \tau) + \text{Noise}$
# 
# - Dao động thật: Cos 80Hz và sin 50Hz
# - Noise: ngẫu nhiên, che khuất dao động chính
# - Lấy mẫu 512 điểm: $t = 0, \frac{1}{512}, \frac{2}{512}, ..., \frac{511}{512}$
# 
# 
# ### Thực hiện biến đổi Fourier rời rạc
# 
# - Gọi $x \in \mathbb{R}^{512}$ là vector tín hiệu thu được.
# - Tính DFT:
#   $y = \frac{2}{n} F_n x = a + ib$
#   Trong đó:
#   - $a = \text{Re}(y)$: biểu diễn phần cos
#   - $b = \text{Im}(y)$: biểu diễn phần sin
# - Chỉ xét nửa đầu phổ (0 → 255) do tính đối xứng.
# 

# In[7]:


# 1. Sinh dữ liệu tín hiệu mẫu
n = 512           # số mẫu
T = 1.0           # thời lượng (giây)
t = np.linspace(0, T, n, endpoint=False)
f1 = 80           # tần số 1 (Hz)
f2 = 50           # tần số 2 (Hz)

np.random.seed(42)
noise = np.random.normal(0, 1, n)
y = np.cos(2 * np.pi * f1 * t) + 2 * np.sin(2 * np.pi * f2 * t) + noise


# In[8]:


# Time domain
plt.figure(figsize=(10,3))
plt.plot(t, y)
plt.title('Composite Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
st.pyplot(plt.gcf())


# In[9]:


Y = dft(y)
freq = np.fft.fftfreq(n, d=t[1] - t[0])

plt.figure(figsize=(10,6))
plt.subplot(2, 1, 1)
plt.stem(freq[:n//2], np.real(Y[:n//2]), basefmt=" ")
plt.title('Frequency Spectrum - Real Part')
plt.ylabel('Amplitude (Real)')

plt.subplot(2, 1, 2)
plt.stem(freq[:n//2], np.imag(Y[:n//2]), basefmt=" ")
plt.title('Frequency Spectrum - Imaginary Part')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (Imag)')
plt.tight_layout()
st.pyplot(plt.gcf())


# In[10]:


y_rec = idft(Y) # Chuyển từ frequency domain -> time domain

plt.figure(figsize=(10,3))
plt.plot(t, y.real, label='Original Signal')
plt.plot(t, y_rec.real, '--', label='Reconstructed from IDFT')
plt.title("Comparison of Original Signal and Reconstructed Signal from IDFT")
plt.xlabel('Time (s)')
plt.legend()
st.pyplot(plt.gcf())


# 
# ### Nhận xét kết quả
# 
# - Phổ phần thực (a): spike tại index 80 $\Rightarrow \cos(2\pi \cdot 80\tau)$
# - Phổ phần ảo (b): spike tại index 50 $\Rightarrow \sin(2\pi \cdot 50\tau)$
# 
# $\Rightarrow \text{Tín hiệu ban đầu đúng là: } y(\tau) = \cos(2\pi 80\tau) + 2 \sin(2\pi 50\tau) + \text{Noise}$
# 
# 
# ### Kết luận
# 
# - DFT giúp tách tần số chính khỏi nhiễu ngẫu nhiên.
# - Biểu đồ phổ tần số cho thấy rõ những thông tin không nhìn được trong thời gian.
# - Cách dùng DFT: $x \rightarrow F_n x \rightarrow \text{Trích xuất dao động}$
# - Biến đổi DFT giúp chuyển tín hiệu từ thời gian sang tần số.
# - Nếu biết DFT, ta có thể “nhìn thấy” những dao động đang bị nhiễu che lấp.
# 

# ### 🔊 Real-world Noise Reduction with DFT

# #### Load and Visualize Audio Signal

# **Các hàm cần thiết**

# In[11]:


# Class vẽ file âm thanh theo 2 miền (Time Domain - Frequency Domain)
class SignalAnalyzer:
    def __init__(self, signal, sr):
        self.signal = signal
        self.sr = sr
        self.n = len(signal)
        self.time_axis = np.linspace(0, self.n / sr, self.n, endpoint=False)
        self.fft_result = np.fft.fft(signal)
        self.frequencies = np.fft.fftfreq(self.n, d=1/sr)

    def amplitude(self):
        return np.abs(self.fft_result)

    def plot_spectrum(self, interactive=False):
        if interactive:
            trace = go.Scatter(x=self.frequencies[:self.n//2],
                               y=self.amplitude()[:self.n//2],
                               mode='lines')
            layout = go.Layout(title='Spectrum',
                               xaxis=dict(title='Frequency [Hz]'),
                               yaxis=dict(title='Amplitude'))
            fig = go.Figure(data=[trace], layout=layout)
            fig.show()
        else:
            plt.figure(figsize=(10,6))
            plt.plot(self.frequencies[:self.n//2], self.amplitude()[:self.n//2])
            plt.title('Spectrum')
            plt.ylabel('Amplitude')
            plt.xlabel('Frequency [Hz]')
            plt.grid(True)
            st.pyplot(plt.gcf())

    def plot_time_frequency(self, t_ylabel="Amplitude", f_ylabel="Amplitude",
                            t_title="Signal (Time Domain)",
                            f_title="Spectrum (Frequency Domain)"):
        # Time domain
        trace_time = go.Scatter(x=self.time_axis, y=self.signal, mode='lines')
        layout_time = go.Layout(title=t_title,
                                xaxis=dict(title='Time [sec]'),
                                yaxis=dict(title=t_ylabel),
                                width=1000, height=400)
        fig_time = go.Figure(data=[trace_time], layout=layout_time)
        fig_time.show()

        # Frequency domain
        trace_freq = go.Scatter(x=self.frequencies[:self.n//2],
                                y=self.amplitude()[:self.n//2],
                                mode='lines')
        layout_freq = go.Layout(title=f_title,
                                xaxis=dict(title='Frequency [Hz]'),
                                yaxis=dict(title=f_ylabel),
                                width=1000, height=400)
        fig_freq = go.Figure(data=[trace_freq], layout=layout_freq)
        fig_freq.show()


#Top frequencies
def analyze_frequency(input_file, start_time, end_time, sr):
    """
    Analyze the frequency components of a signal in a specific time range.

    Parameters:
        input_file (str): Path to the input audio file (.wav).
        start_time (float): Start time in seconds for analysis.
        end_time (float): End time in seconds for analysis.
        sr (int): Sampling rate of the audio file (Hz).

    Returns:
        freq_peaks (list): List of frequencies with the highest amplitudes.
    """
    # Step 1: Load the audio file
    y_real, sr = sf.read(input_file)
    if len(y_real.shape) > 1:  # Convert to mono if stereo
        y_real = np.mean(y_real, axis=1)

    # Step 2: Define the sample range for the desired time window
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    signal_segment = y_real[start_sample:end_sample]

    # Step 3: Apply FFT to the segment
    n = len(signal_segment)
    signal_fft = np.fft.fft(signal_segment)
    freqs = np.fft.fftfreq(n, d=1/sr)
    magnitude = np.abs(signal_fft)

    # Step 4: Filter positive frequencies only
    pos_freqs = freqs[:n // 2]
    pos_magnitude = magnitude[:n // 2]

    # Step 5: Find the peaks in the frequency domain
    peak_indices = np.argsort(pos_magnitude)[-5:]  # Get the top 5 frequencies
    freq_peaks = pos_freqs[peak_indices]

    # Step 6: Plot the spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(pos_freqs, pos_magnitude)
    plt.title("Spectrum (Frequency Domain)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.axvline(x=freq_peaks[0], color='r', linestyle='--', label=f"Peak: {freq_peaks[0]:.2f} Hz")
    plt.legend()
    plt.grid()
    st.pyplot(plt.gcf())

    print(f"Top frequencies in the range {start_time}-{end_time} seconds:", freq_peaks)
    return freq_peaks



# **Load file**

# In[12]:


file_path = "D:/Download/noisy_sound.mp3"


# In[13]:


audio = AudioSegment.from_file(file_path)
if audio.channels > 1:
    audio = audio.set_channels(1)
#----------------------------------------------------------------------#
signal_vector = np.array(audio.get_array_of_samples(), dtype=np.float64)
frame_rate = audio.frame_rate
sample_width = audio.sample_width
channels = audio.channels
display(Audio(signal_vector, rate=frame_rate))
print(f"Tín hiệu đã được tạo với kích thước: {signal_vector.shape[0]}")
print(f"Tốc độ khung hình (frame rate): {frame_rate} Hz")
print(f"Độ rộng mẫu (sample width): {sample_width} bytes")
print(f"Số kênh (channels): {channels}")


# **Time Domain vs. Frequency Domain**

# In[ ]:


analyzer = SignalAnalyzer(signal_vector, frame_rate)
analyzer.plot_time_frequency()        # Vẽ tương tác cả 2 miền


# **Xác định các miền tần trong khoảng thời gian nhất định**

# In[ ]:


start_time = 250  # Start time in seconds
end_time = 289  # End time in seconds
top_frequencies = analyze_frequency(file_path , start_time, end_time, frame_rate)


# **Chạy DFT**

# In[14]:


# Y = dft(signal_vector)


# > Vì dùng dft thủ công thì bộ nhớ không cấp phát nổi → cái fft giúp giảm độ phức tạp xuống

# ### 🚫 Common DFT Issues: Aliasing, Leakage, Resolution

# #### Aliasing

# In[ ]:





# #### Leakage

# In[ ]:





# #### Resolution

# In[ ]:





# ### ⚡ From DFT to FFT

# Vì thuật toán DFT truyền thống đòi hỏi lượng lớn phép tính và bộ nhớ, đặc biệt với tín hiệu dài, nên người ta đã phát triển phương pháp phân rã ma trận Fourier từ đó tạo ra thuật toán FFT, giúp tăng tốc tính toán đáng kể.

# #### Decomposing the Fourier Matrix
# 
# Giúp tính toán Fourier Transform nhanh hơn bằng cách **phân rã ma trận Fourier lớn** $\mathbf{F}_n$ thành những phần nhỏ hơn, cụ thể là 2 ma trận Fourier cấp $\frac{n}{2}$.  
# 
# Nếu $n = 2^r$, thì:
# $
# \mathbf{F}_n =
# \begin{pmatrix}
# \mathbf{F}_{n/2} & \mathbf{D}_{n/2} \mathbf{F}_{n/2} \\
# \mathbf{F}_{n/2} & -\mathbf{D}_{n/2} \mathbf{F}_{n/2}
# \end{pmatrix}
# \mathbf{P}_n
# $
# 
# Trong đó:
# 
# - $\mathbf{F}_n$: Ma trận Fourier cấp **n**.  
# - $\mathbf{F}_{n/2}$: Ma trận Fourier cấp nhỏ hơn, kích thước $\frac{n}{2}$.  
# - $\mathbf{D}_{n/2}$: Ma trận đường chéo chứa các căn bậc **n** của 1.  
# - $\mathbf{P}_n$: Ma trận hoán vị "even–odd" (đổi vị trí chỉ số chẵn và lẻ).
# 
# 
# 
# **Ma trận $\mathbf{D}_{n/2}$:**
# 
# $\mathbf{D}_{n/2} = \mathrm{diag}(1, \xi, \xi^2, \dots, \xi^{n/2 - 1})$
# 
# - Với $\xi = e^{-2\pi i / n}$.  
# 
# 
# **Ma trận hoán vị $\mathbf{P}_n$:**
# 
# $
# \mathbf{P}_n^T = [e_0, e_2, e_4, \dots, e_{n-2} \mid e_1, e_3, e_5, \dots, e_{n-1}]
# $
# 
# - Vector cơ sở được sắp xếp lại:
#   - Các chỉ số **chẵn** đứng trước.
#   - Các chỉ số **lẻ** đứng sau.
# - Tác dụng: tách tín hiệu đầu vào thành **even** và **odd**, phục vụ cho bước chia để trị.
# 
# 

# #### Biến đổi Fourier Nhanh (Fast Fourier Transform)
# 
# Với một vector đầu vào **x** có số phần tử $n = 2^r$, biến đổi Fourier rời rạc $\mathbf{F}_n \mathbf{x}$ được tính bằng cách lần lượt tạo ra các mảng sau:
# 
# $
# \mathbf{X}_{1 \times n} \leftarrow \text{rev}(\mathbf{x}) \quad \text{(đảo bit các chỉ số)}
# $
# 
# Với $j = 0, 1, 2, 3, \ldots, r - 1$:
# 
# $
# \mathbf{D} \leftarrow
# \begin{pmatrix}
# 1 \\
# e^{-\pi i / 2^j} \\
# e^{-2\pi i / 2^j} \\
# e^{-3\pi i / 2^j} \\
# \vdots \\
# e^{-(2^j - 1)\pi i / 2^j}
# \end{pmatrix}_{2^j \times 1}$
# Một nửa số căn bậc $2^{j+1}$ của 1, có thể lấy từ bảng tra trước
# 
# $
# \mathbf{X}^{(0)} = 
# \begin{pmatrix}
# \mathbf{X}_{*0} \quad \mathbf{X}_{*2} \quad \mathbf{X}_{*4} \quad \cdots \quad \mathbf{X}_{*2^{r - j} - 2}
# \end{pmatrix}_{2^j \times 2^{r - j - 1}}
# $
# 
# $
# \mathbf{X}^{(1)} = 
# \begin{pmatrix}
# \mathbf{X}_{*1} \quad \mathbf{X}_{*3} \quad \mathbf{X}_{*5} \quad \cdots \quad \mathbf{X}_{*2^{r - j} - 1}
# \end{pmatrix}_{2^j \times 2^{r - j - 1}}
# $
# 
# $
# \mathbf{X} \leftarrow
# \begin{pmatrix}
# \mathbf{X}^{(0)} + \mathbf{D} \times \mathbf{X}^{(1)} \\
# \mathbf{X}^{(0)} - \mathbf{D} \times \mathbf{X}^{(1)}
# \end{pmatrix}_{2^{j+1} \times 2^{r - j - 1}}
# $
# 
# > Định nghĩa phép nhân phần tử:  
# > $
# > [\mathbf{D} \times \mathbf{M}]_{ij} = d_i m_{ij}
# > $
# 
# 

# In[ ]:


def fft_from_scratch(x):
    """
    FFT đệ quy thuần lý thuyết, theo phân rã ma trận Fourier:
    F_n = [[F_{n/2}, D*F_{n/2}],
           [F_{n/2}, -D*F_{n/2}]] * P_n
    """
    x = np.asarray(x, dtype=complex)
    n = x.shape[0]

    if n == 1:
        return x

    if n & (n - 1) != 0:
        raise ValueError("Input length must be a power of 2")

    # Đệ quy chia để trị
    even = fft_from_scratch(x[::2])
    odd = fft_from_scratch(x[1::2])

    # Xoay pha
    twiddles = np.exp(-2j * np.pi * np.arange(n // 2) / n)

    # Kết hợp theo công thức phân rã
    top = even + twiddles * odd
    bottom = even - twiddles * odd

    return np.concatenate([top, bottom])


# #### **2. Biến đổi Fourier Nhanh Ngược (IFFT) - Dựa trên IDFT**

# 
# Giả sử $\mathbf{X}$ đã được đảo bit thứ tự (bit-reversal), thì thực hiện theo từng mức $j = 0, 1, \dots, r-1$:
# 
# Tạo vector xoay pha (liên hợp của vector FFT):
# 
# $
# \mathbf{D} \leftarrow
# \begin{pmatrix}
# 1 \
# e^{+\pi i / 2^j} \
# e^{+2\pi i / 2^j} \
# e^{+3\pi i / 2^j} \
# \vdots \
# e^{+(2^j - 1)\pi i / 2^j}
# \end{pmatrix}_{2^j \times 1}
# $
# 
# Tách ma trận đầu vào thành 2 nửa:
# 
# $
# \mathbf{X}^{(0)} =
# \begin{pmatrix}
# \mathbf{X}{*0} \quad \mathbf{X}{*2} \quad \mathbf{X}{*4} \quad \cdots
# \end{pmatrix}
# \quad,\quad
# \mathbf{X}^{(1)} =
# \begin{pmatrix}
# \mathbf{X}{*1} \quad \mathbf{X}_{*3} \quad \cdots
# \end{pmatrix}
# $
# 
# Cập nhật $\mathbf{X}$ ở bước tiếp theo:
# 
# $
# \mathbf{X} \leftarrow
# \begin{pmatrix}
# \mathbf{X}^{(0)} + \mathbf{D} \times \mathbf{X}^{(1)} \
# \mathbf{X}^{(0)} - \mathbf{D} \times \mathbf{X}^{(1)}
# \end{pmatrix}
# $
# 
# Sau cùng, chia toàn bộ $\mathbf{X}$ cho $n$ để chuẩn hóa:
# 
# $
# \boxed{
# \mathbf{x} = \frac{1}{n} \cdot \mathbf{X}
# }
# $

# In[ ]:


# ifft dùng idft
def ifft_from_definition(X):
    """
    IFFT từ định nghĩa không dùng FFT, không dùng liên hợp.
    """
    X = np.asarray(X, dtype=complex)
    n = X.shape[0]
    if n & (n - 1) != 0:
        raise ValueError("Input length must be a power of 2")

    F_inv = idft(n)
    x = F_inv @ X
    return x.real  # Chỉ lấy phần thực nếu là tín hiệu âm thanh


# #### **3. Biến đổi Fourier Nhanh Ngược (IFFT) - Dựa trên FFT**

# IFFT (Inverse Fast Fourier Transform) là thuật toán hiệu quả để tính toán Biến đổi Fourier Rời rạc Ngược (IDFT). Thay vì triển khai một thuật toán IFFT riêng biệt từ đầu, một phương pháp phổ biến và hiệu quả là tái sử dụng các thuật toán FFT (thuận) đã có. Phương pháp này dựa trên một tính chất quan trọng của DFT và IDFT.
# 
# Mối quan hệ giữa DFT và IDFT có thể được khai thác để tính toán IDFT bằng cách sử dụng một thuật toán DFT (FFT) chuẩn. Công thức của IDFT là:
# 
# $$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k]e^{+j\frac{2\pi}{N}nk}$$
# 
# Công thức của DFT là:
# 
# $$Y[k] = \sum_{n=0}^{N-1} y[n]e^{-j\frac{2\pi}{N}nk}$$
# 
# Nếu chúng ta so sánh hai công thức này, ta có thể thấy một mối liên hệ thông qua toán tử liên hợp phức (conjugate).
# Cụ thể, nếu chúng ta tính DFT của liên hợp phức của $X[k]$ (tức là $\bar{X}[k]$):
# 
# $$DFT(\bar{X}[k]) = \sum_{k=0}^{N-1} \bar{X}[k]e^{-j\frac{2\pi}{N}nk}$$
# 
# Bằng cách sử dụng tính chất $\overline{(AB)} = \bar{A}\bar{B}$ và $\overline{(e^{j\theta})} = e^{-j\theta}$, ta có:
# 
# $$DFT(\bar{X}[k]) = \overline{\left( \sum_{k=0}^{N-1} X[k]e^{+j\frac{2\pi}{N}nk} \right)}$$
# 
# Nhận thấy rằng tổng trong ngoặc đơn chính là $N \cdot x[n]$ từ công thức IDFT:
# 
# $$DFT(\bar{X}[k]) = \overline{(N \cdot x[n] )} = N \cdot \bar{x}[n]$$
# 
# Từ đó, suy ra:
# $$\bar{x}[n] = \frac{1}{N} DFT(\bar{X}[k])$$
# 
# Và để tìm $x[n]$, chúng ta chỉ cần lấy liên hợp phức của kết quả này:
# 
# $$x[n] = \overline{\left( \frac{1}{N} DFT(\bar{X}[k]) \right)} = \frac{1}{N} \overline{DFT(\bar{X}[k])}$$

# In[ ]:


def ifft_from_scratch(X):
    """
    IFFT đệ quy theo định nghĩa:
    ifft(x) = conj(fft(conj(x))) / n
    """
    X = np.asarray(X, dtype=complex)
    n = X.shape[0]

    if n & (n - 1) != 0:
        raise ValueError("Input length must be a power of 2")

    x = fft_from_scratch(np.conjugate(X))
    x = np.conjugate(x) / n # conjugate() tính liên hợp phức

    return x.real


# In[ ]:





# #### 🔬 Convolution trong FFT

# 
# Cho  $\mathbf{a} \times \mathbf{b}$ là phép **nhân từng phần tử tương ứng**, ta có:
# 
# $\mathbf{a} \times \mathbf{b} =
# \begin{pmatrix}
# \alpha_0 \\
# \alpha_1 \\
# \vdots \\
# \alpha_{n-1}
# \end{pmatrix} \times \begin{pmatrix}
# \beta_0 \\
# \beta_1 \\
# \vdots \\
# \beta_{n-1}
# \end{pmatrix}
# =$
# $
# \begin{pmatrix}
# \alpha_0 \beta_0 \\
# \alpha_1 \beta_1 \\
# \vdots \\
# \alpha_{n-1} \beta_{n-1}
# \end{pmatrix}_{n \times 1}$
# 
# 
# 
# Giả sử $\hat{\mathbf{a}}$ và $\hat{\mathbf{b}}$ là **các vector được padding thêm 0** để có độ dài gấp đôi:
# 
# $
# \hat{\mathbf{a}} =
# \begin{pmatrix}
# \alpha_0 \\
# \vdots \\
# \alpha_{n-1} \\
# 0 \\
# \vdots \\
# 0
# \end{pmatrix}_{2n \times 1}
# \quad \text{và} \quad
# \hat{\mathbf{b}} =
# \begin{pmatrix}
# \beta_0 \\
# \vdots \\
# \beta_{n-1} \\
# 0 \\
# \vdots \\
# 0
# \end{pmatrix}_{2n \times 1}
# $
# 
# Nếu $\mathbf{F} = \mathbf{F}_{2n}$ là **ma trận Fourier rời rạc cấp $2n$** (áp dụng trong FFT), thì:
# $
# \mathbf{F}(\mathbf{a} * \mathbf{b}) = (\mathbf{F} \hat{\mathbf{a}}) \times (\mathbf{F} \hat{\mathbf{b}})
# $ 
# và:
# $
# \boxed{
# \mathbf{a} * \mathbf{b} = \mathbf{F}^{-1} \left[ (\mathbf{F} \hat{\mathbf{a}}) \times (\mathbf{F} \hat{\mathbf{b}}) \right]
# }
# \quad \text{(5.8.12)}
# $
# 
# 
# 
# > **Tích chập tuyến tính trong miền thời gian** có thể được thực hiện hiệu quả hơn bằng cách **chuyển sang miền tần số**, nhờ định lý tích chập.
# 
# Thay vì tính trực tiếp tích chập (rất tốn thời gian), ta làm như sau:
# 
# 1. **Chèn thêm số 0 (padding)** vào tín hiệu và bộ lọc để tránh hiện tượng wrap-around.
# 2. **Dùng FFT** để biến đổi cả hai sang miền tần số:  
#    $$
#    X = \text{FFT}(x), \quad H = \text{FFT}(h)
#    $$
# 3. **Nhân từng phần tử tương ứng trong miền tần số**:  
#    $$
#    Y = X \cdot H
#    $$
# 4. **Dùng IFFT** để đưa kết quả trở lại miền thời gian:  
#    $$
#    y = \text{IFFT}(Y)
#    $$
# 
# 

# Time domain → FFT → Nhân → IFFT → Filtered signal

# In[ ]:


def convolution_filter(signal_fft, kernel_fft):
    """
    Perform convolution in the frequency domain.
    signal_fft: array-like, FFT of the signal.
    kernel_fft: array-like, FFT of the kernel/filter.
    return: array-like, FFT of the filtered signal.
    """
    return signal_fft * kernel_fft

def filter_audio(input_file, output_file, cutoff_freq, sr, filter_type='low-pass'):
    """
    Filter an audio file using FFT and convolution, supporting both low-pass and high-pass filters.
    input_file: str - Path to the input audio file (.wav).
    output_file: str - Path to save the filtered audio file (.wav).
    cutoff_freq: float - Cutoff frequency for the filter (Hz).
    sr: int - Sampling rate of the audio file (Hz).
    filter_type: str - Type of filter ('low-pass', 'high-pass', or 'band-pass').
    """
    # Step 1: Load the audio file
    y_real, sr = sf.read(input_file)
    if len(y_real.shape) > 1:  # Convert to mono if stereo
        y_real = np.mean(y_real, axis=1)

    # Ensure signal length is sufficient for FFT
    n = len(y_real)
    fft_size = 2 ** int(np.ceil(np.log2(n + 1)))  # Ensure enough padding for convolution
    y_real = np.pad(y_real, (0, fft_size - n), mode='constant')

    # Step 2: Apply FFT to signal
    signal_fft = fft_from_scratch(y_real)

    # Step 3: Create kernel/filter in the frequency domain
    freqs = np.fft.fftfreq(fft_size, d=1/sr)
    kernel = np.zeros(fft_size, dtype=complex)
    if filter_type == 'low-pass':
        kernel[np.abs(freqs) <= cutoff_freq] = 1  # Low-pass filter
    elif filter_type == 'high-pass':
        kernel[np.abs(freqs) >= cutoff_freq] = 1  # High-pass filter
    elif filter_type == 'band-pass':
        bandwidth = 10  # Example bandwidth (adjustable)
        kernel[(np.abs(freqs) >= cutoff_freq - bandwidth) & (np.abs(freqs) <= cutoff_freq + bandwidth)] = 1
    else:
        raise ValueError("Invalid filter_type. Use 'low-pass', 'high-pass', or 'band-pass'.")

    kernel_fft = fft_from_scratch(kernel)

    # Step 4: Apply convolution in the frequency domain
    filtered_fft = convolution_filter(signal_fft, kernel_fft)

    # Step 5: Apply IFFT to get the filtered signal
    print("Performing IFFT...")
    filtered_signal = ifft_from_scratch(filtered_fft)

    # Step 6: Normalize the filtered signal
    filtered_signal = filtered_signal[:n]  # Remove padding
    filtered_signal = np.int16(filtered_signal / np.max(np.abs(filtered_signal)) * 32767)  # Normalize to 16-bit

    # Step 7: Save the filtered audio
    write(output_file, sr, filtered_signal)
    print(f"Filtered audio saved to {output_file}")

    # Return the filtered signal for playback
    return filtered_signal


# Phần này nó chạy bị lặp âm thanh t đang kiếm lý do tại sao

# In[39]:


# output_path = "mixed_convolution.wav"  # Output audio file path
# cutoff = 287  # Cutoff frequency in Hz
# filtered_audio = filter_audio(file_path, output_path, cutoff, frame_rate, filter_type='low-pass')


# # In[40]:


# # Play the filtered audio
# Audio(filtered_audio , rate=frame_rate)


# #### Sử dụng fftconvolve kết hợp ngưỡng, Gaussian (sử dụng thư viện)

# In[36]:


def filter_combined(input_file, output_file, freq_range, start_time, end_time, threshold=0.4, kernel_size=50, kernel_width=2.0, play_audio=False):
    """
    Apply threshold, Band-Stop Filter, and Gaussian smoothing to filter abnormal signal.

    Parameters:
        input_file (str): Path to the input audio file (.wav).
        output_file (str): Path to save the filtered audio file (.wav).
        freq_range (tuple): Frequency range to be filtered (low_freq, high_freq) in Hz.
        start_time (float): Start time of the range to apply the filter (in seconds).
        end_time (float): End time of the range to apply the filter (in seconds).
        threshold (float): Amplitude threshold to filter abnormal values.
        kernel_size (int): Size of the Gaussian kernel.
        kernel_width (float): Width of the Gaussian kernel.
        play_audio (bool): Whether to play the audio after filtering.
    """
    # Load audio file
    signal, sr = sf.read(input_file)
    if len(signal.shape) > 1:  # Convert to mono if stereo
        signal = np.mean(signal, axis=1)

    # Convert time range to sample range
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Step 1: Apply threshold to specific time range
    segment = signal[start_sample:end_sample]
    segment[segment > threshold] = threshold  # Cap values above threshold
    segment[segment < -threshold] = -threshold  # Cap values below threshold

    # Step 2: Apply Band-Stop Filter to the thresholded segment
    low_freq, high_freq = freq_range
    nyquist = sr / 2  # Nyquist frequency
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(N=2, Wn=[low, high], btype='bandstop')  # Create Band-Stop Filter coefficients
    filtered_segment = filtfilt(b, a, segment)  # Apply Band-Stop Filter

    # Replace the original segment with the filtered segment
    signal[start_sample:end_sample] = filtered_segment

    # Step 3: Apply Gaussian smoothing to the entire signal
    kernel = np.exp(-np.linspace(-kernel_width, kernel_width, kernel_size)**2)
    kernel = kernel / np.sum(kernel)  # Normalize
    smoothed_signal = fftconvolve(signal, kernel, mode='same')  # Apply Gaussian smoothing

    # Normalize and save filtered audio
    smoothed_signal = smoothed_signal / np.max(np.abs(smoothed_signal))  # Normalize
    sf.write(output_file, smoothed_signal, sr)
    print(f"Filtered audio saved to {output_file}")

    # Plot signals for comparison
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(signal, label='Original Signal')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(filtered_segment, label=f'Filtered Segment @ {freq_range} Hz with Threshold {threshold}', color='orange')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(smoothed_signal, label='Smoothed Signal (Gaussian Kernel)', color='green')
    plt.legend()
    st.pyplot(plt.gcf())

    # Play audio if requested
    if play_audio:
        print("Playing filtered audio...")
        return Audio(data=smoothed_signal, rate=sr)



# In[37]:


output_path = "filtered_combined_signal.wav"

# Apply combined filter to remove frequency range (80 Hz to 90 Hz) with threshold
audio_player = filter_combined(
    input_file=file_path,
    output_file=output_path,
    freq_range=(92,278),  # Frequency range to remove
    start_time=22,        # Start time (seconds)
    end_time=23,          # End time (seconds)
    threshold= 1,        # Amplitude threshold
    kernel_size=50,       # Size of Gaussian kernel
    kernel_width=3.0,     # Width of Gaussian kernel
    play_audio=True
)

# Display audio player if returned
if audio_player:
    display(audio_player)


# ###  FFT lọc theo magnitude threshold

# In[41]:


data_fft = np.fft.fft(signal_vector)
full_magnitudes = np.abs(data_fft)
max_magnitude = np.max(full_magnitudes)


# In[42]:


N_output = len(data_fft)  # Tổng số điểm trong phổ FFT
delta_f = frame_rate / N_output   # Độ phân giải tần số (bước nhảy giữa các bin)
num_positive_bins = (N_output + 1) // 2 # Số lượng bin tần số dương (bao gồm DC)
frequencies = np.zeros(num_positive_bins, dtype=np.float64)
# Tính toán tần số dương
for k in range(num_positive_bins):
    frequencies[k] = k * delta_f
#---------------------------------------------------------------------------------#
magnitudes = np.abs(data_fft[:num_positive_bins]) # Lấy biên độ thô của phần dương
# Chuẩn hóa ban đầu: Chia cho tổng số điểm FFT
magnitudes = magnitudes / N_output
# Nhân đôi các thành phần để lấy biên độ vật lý (trừ DC và Nyquist)
if N_output % 2 == 0:
    magnitudes[1:-1] = magnitudes[1:-1] * 2
else:
    magnitudes[1:] = magnitudes[1:] * 2
threshold = 0.015
threshold_value = threshold * max_magnitude
filtered = np.where(np.abs(data_fft) > threshold_value, data_fft, 0)
#---------------------------------------------------------------------------------#
filtered_magnitudes = np.abs(filtered[:num_positive_bins])
# Chuẩn hóa ban đầu: Chia cho tổng số điểm FFT
filtered_magnitudes = filtered_magnitudes / N_output
# Nhân đôi các thành phần để lấy biên độ vật lý (trừ DC và Nyquist)
if N_output % 2 == 0:
    filtered_magnitudes[1:-1] = filtered_magnitudes[1:-1] * 2
else:
    filtered_magnitudes[1:] = filtered_magnitudes[1:] * 2
#---------------------------------------------------------------------------------#
plt.figure(figsize=(14.5, 6))
# Vẽ phổ tần số gốc với màu xanh lam
plt.plot(frequencies, magnitudes,
        label='Phổ Gốc', color='#1f77b4', linewidth=1)
# Vẽ phổ tần số đã làm sạch với màu cam, đè lên phổ gốc
plt.plot(frequencies, filtered_magnitudes,
        label='Phổ Đã Làm Sạch', color='#ff7f0e', linewidth=1)
plt.title(f'Phổ Tần Số của Tín hiệu Âm thanh (Gốc và Đã Làm Sạch)')
plt.xlabel('Tần số (Hz)')
plt.ylabel('Biên độ')
plt.grid(True)
plt.xlim(0, frame_rate / 2)
plt.legend()
st.pyplot(plt.gcf())


# In[43]:


signal_fft_inverse = np.real(np.fft.ifft(filtered)).astype(np.float64)


# **Nghe lại tín hiệu đã lọc**

# In[44]:


# Tạo đối tượng AudioSegment.
audio_segment = AudioSegment(
    signal_fft_inverse.astype(np.int16).tobytes(),
    frame_rate=frame_rate,
    sample_width=sample_width,
    channels=channels
)
audio_segment += 2
display(audio_segment)
#---------------------------------------------------------------------------------#
time_axis_original = np.arange(len(signal_vector)) / frame_rate
time_axis_cleaned = np.arange(len(signal_fft_inverse)) / frame_rate

plt.figure(figsize=(12, 6))
# Vẽ tín hiệu gốc với màu xanh lam
plt.plot(time_axis_original, signal_vector,
            label='Tín hiệu Gốc', color='#1f77b4', linewidth=1)
# Vẽ tín hiệu đã làm sạch với màu cam
plt.plot(time_axis_cleaned, signal_fft_inverse,
            label='Tín hiệu Đã Làm Sạch', color='#ff7f0e', linewidth=1)
plt.title('Tín hiệu Âm thanh trong Miền Thời gian (Gốc và Đã Làm Sạch)')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Biên độ')
plt.grid(True)
plt.legend()
plt.legend(loc='upper left')
plt.tight_layout()
st.pyplot(plt.gcf())


# #### Khử nhiễu bằng phương pháp trừ phổ (SPECTRAL SUBTRACTION)

# In[ ]:





# #### Khử nhiễu bằng ngưỡng mật độ phổ công suất (PSD THRESHOLDING) trực tiếp

# In[ ]:





# #### ANN 

# In[ ]:


# def relu(Z):
#     return np.maximum(0, Z)

# def relu_derivative(Z):
#     return Z > 0

# def forward_prop(W1, W2, X):
#     Z1 = np.dot(W1, X)
#     A1 = relu(Z1)
#     Z2 = np.dot(W2, A1)
#     A2 = Z2
#     return Z1, A1, Z2, A2

# def back_prop(m, W1, W2, Z1, A1, Z2, A2, Y):
#     dZ2 = (A2 - Y)
#     dW2 = np.dot(dZ2, A1.T) / m
#     dA1 = np.dot(W2.T, dZ2)
#     dZ1 = dA1 * relu_derivative(Z1)
#     dW1 = np.dot(dZ1, X.T) / m
#     return dZ2, dW2, dZ1, dW1

# N = 32
# batch = 10000
# lr = 0.01
# iterations = 10000

# sig = np.random.randn(batch, N) + 1j*np.random.randn(batch, N)
# F = np.fft.fft(sig, axis=-1)
# X = np.hstack([sig.real, sig.imag]).T
# Y = np.hstack([F.real, F.imag]).T
# n_input = X.shape[0]
# n_output = Y.shape[0]
# n_hidden = 128


# np.random.seed(42)
# W1 = np.random.randn(n_hidden, n_input) * 0.01
# W2 = np.random.randn(n_output, n_hidden) * 0.01

# losses = []
# m = X.shape[1]
# for i in range(iterations):
#     Z1, A1, Z2, A2 = forward_prop(W1, W2, X)
#     loss = np.mean((A2 - Y) ** 2)
#     losses.append(loss)
#     dZ2, dW2, dZ1, dW1 = back_prop(m, W1, W2, Z1, A1, Z2, A2, Y)
#     W2 -= lr * dW2
#     W1 -= lr * dW1
#     if i % 1000 == 0:
#         print(f"Iteration {i}: Loss = {loss:.6f}")

# plt.plot(losses)
# plt.xlabel("Epochs")
# plt.ylabel("Loss (MSE)")
# plt.yscale('log')
# plt.title("Training Loss (DFT Approximation)")
# st.pyplot(plt.gcf())


# ##  Conclusion

# In[ ]:





# References:
# 
# [1] Meyer C.D, Matrix analysis and Applied linear algebra, SIAM, 2000, chapter 5, section 8
# 
# [2] Isaac Amidror, Mastering the Discrete Fourier Transform in One, Two or Several Dimensions: Pitfalls and Artifacts, Springer,2013
