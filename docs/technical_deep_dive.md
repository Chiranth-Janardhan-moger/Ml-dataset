# The Silent Cry Decoder: Technical Deep-Dive

This document provides a technical deep-dive into the signal processing math, deep learning architectures, fusion methodologies, and software design patterns used in **The Silent Cry Decoder** project.

---

## 1. Digital Signal Processing (DSP) & Feature Engineering

Before audio waveforms are fed into neural networks, they must be converted from the time domain to representation domains that expose key spectral features.

### A. Mel-Spectrogram (Visual Stream Input)
The Mel-Spectrogram transforms the raw 1D time-series pressure waves into a 2D time-frequency representation that mirrors human hearing.
1.  **Short-Time Fourier Transform (STFT)**: Splits the continuous signal into overlapping frames (frame size $N = 2048$, hop size $H = 512$). For each frame, it computes the Discrete Fourier Transform (DFT):
    $$X(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n - mH] e^{-j \omega n}$$
    where $w[n]$ is a Hann window function applied to prevent spectral leakage.
2.  **Mel-Scale Warping**: The physical frequency $f$ in Hertz is mapped to the perceptual Mel scale $m$ via:
    $$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$
    This compresses high-frequency regions (where humans are less sensitive to pitch changes) and expands low-frequency regions. We apply 128 triangular Mel filterbanks to group DFT bins.
3.  **Logarithmic Scaling**: Convert power to decibels (dB) via $S_{dB} = 10 \log_{10}(P/P_{\text{ref}})$. Baby cries are highly dynamic, and log scaling matches logarithmic human perception of loudness.

### B. Handcrafted Sequential Features (BiLSTM Input)
1.  **MFCCs (Mel-Frequency Cepstral Coefficients)**: Computed by taking the Discrete Cosine Transform (DCT) of the log-filterbank energies. Retaining the first 40 coefficients captures the envelope of the short-time power spectrum (the "timbre" or vocal tract shape of the baby).
2.  **Deltas & Delta-Deltas**: First-order (velocity) and second-order (acceleration) derivatives of the MFCCs. They capture spectral trajectories over time:
    $$d_t = \frac{\sum_{n=1}^{N} n(c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{N} n^2}$$
    where $c_t$ is the MFCC vector at time $t$. Adding deltas increases the feature dimension from 40 to 120, exposing rapid frequency shifts (crying vs. gasping).
3.  **Spectral Centroid**: The "center of gravity" of the spectrum:
    $$f_c = \frac{\sum_{n} f(n) |X(n)|^2}{\sum_{n} |X(n)|^2}$$
    This indicates brightness. Pain cries are highly shrill (high spectral centroid).
4.  **Zero-Crossing Rate (ZCR)**: The rate at which the signal changes sign:
    $$\text{ZCR} = \frac{1}{2(N-1)} \sum_{n=1}^{N-1} |\operatorname{sgn}(x[n]) - \operatorname{sgn}(x[n-1])|$$
    Indicates noise vs. harmonic content. Frictional cries (like discomfort or burping) have higher ZCR due to air turbulence.
5.  **Root-Mean-Square (RMS) Energy**: Represents signal envelope power:
    $$x_{\mathrm{RMS}} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} x[n]^2}$$
    Identifies respiratory cycles and intensity.

---

## 2. Stream Architecture Details

### Stream 1: Visual Stream (EfficientNet-B0)
EfficientNet uses compound scaling to balance depth, width, and resolution using a compound coefficient $\phi$.
*   **MBConv Block**: The core building block is the Mobile Inverted Bottleneck Convolution. It expands the channels using a $1\times1$ expansion conv, applies a depthwise $3\times3$ or $5\times5$ conv (reducing parameter count), applies **Squeeze-and-Excitation (SE)** channel attention, and projects back with a linear $1\times1$ point-wise conv.
*   **Freezing Strategy**: In torchvision's implementation, the features backbone comprises 9 sequential stages. We freeze stages 0 to 5. Stages 6 (4 MBConv blocks), 7 (1 MBConv block), and 8 (final $1\times1$ Conv-BN-Swish) are left trainable, letting the network adapt its high-level shape filters to the structures of Mel-spectrograms.

### Stream 2: Sequential Stream (BiLSTM + Self-Attention)
*   **BiLSTM**: Processes sequences in both forward and backward directions. This is crucial for audio, as future audio frames (e.g. the end of a cry) provide context for past frames (e.g. the build-up phase).
    $$\vec{h}_t = \text{LSTM}_{\text{fwd}}(x_t, \vec{h}_{t-1}), \quad \overleftarrow{h}_t = \text{LSTM}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1})$$
    $$y_t = [\vec{h}_t; \overleftarrow{h}_t]$$
*   **Temporal Self-Attention**: Rather than simple pooling, the self-attention layer weights frames depending on their acoustic density. By computing $Q, K, V$ states, it dynamically focuses on key frames (e.g., sudden shrieks) while suppressing silence.

### Stream 3: Pretrained Stream (Whisper Encoder)
Whisper uses a standard Encoder-Decoder Transformer architecture.
*   **Acoustic Front-end**: Resamples audio to 16kHz and processes it using two convolutional layers with a filter width of 3 and stride of 2, downsampling the sequence length by half.
*   **Sinusoidal Positional Embeddings**: Added to convolutional representations to inject temporal order.
*   **Encoder Blocks**: Consists of 12 Transformer blocks (for `whisper-small`), containing Multi-Head Self-Attention layers followed by Layer Normalization and MLP blocks.
*   **Feature Output**: The encoder outputs representation matrices of shape $(B, 1500, 768)$. Global average pooling collapses the sequence dimension to $(B, 768)$, and a trainable projection head compresses it to the $(B, 128)$ joint embedding space.

---

## 3. Embedding Fusion & Classification Head

### Fusion Method Selection
We select **Late Feature Concatenation** over other fusion schemes:

| Fusion Type | Implementation | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Early Fusion** | Concatenating spectrograms/features at raw input stage. | Single model pipeline. | Streams must have identical temporal alignment; fails with highly divergent features. |
| **Concatenation (Selected)** | Combining 1D embeddings: $[v_i; s_i; w_i]$. | Simple, preserves all features, linear scalability. | Increases classifier head parameter count. |
| **Bilinear Pooling** | Computing Kronecker product: $v_i \otimes s_i \otimes w_i$. | Captures complex cross-stream multiplicative relationships. | Dimensionality explosion ($128^3 = 2,097,152$ dims). Extremely prone to overfitting on small datasets. |

### Classification Architecture
The 384-dim concatenated representation is fed into an MLP:
1.  **Dense 256**: Maps $384 \to 256$ dimensions.
2.  **Batch Normalization**: Stabilizes training by normalizing activations to zero mean and unit variance.
3.  **Dropout (0.4)**: Regularizes the model. During training, 40% of activations are set to zero, forcing the model to develop redundant paths.
4.  **Dense 128**: Maps $256 \to 128$ dimensions.
5.  **Dropout (0.3)**: 30% dropout.
6.  **Dense 5**: Maps $128 \to 5$ logits.

---

## 4. Software Design Patterns & Architecture

### Backend Design Patterns (FastAPI & PyTorch)
*   **Singleton Pattern (Inference Pipeline)**: Model loading (especially downloading Whisper weights and reading CNN tensors) is expensive. We load `InferencePipeline` once on startup inside FastAPI's `@app.on_event("startup")` event handler. The pipeline instance is persisted in memory, ensuring subsequent HTTP requests run inference in sub-100ms.
*   **Factory / Decoupled Feature Processing**: Preprocessing functions in `src/features.py` are stateless utility functions. This decoupling permits training scripts, validation scripts, and production servers to import and use the exact same extraction code, eliminating train-serve skew.
*   **Resource Cleanup Pattern**: The API endpoint receives uploads, writes a UUID-labeled temp file to disk, runs predictions, and guarantees deletion using a `finally` block to protect against disk exhaustion attacks.

### Frontend Architecture (PWA & Web Audio API)
*   **Service Worker Architecture**: Uses a client-side proxy service worker (`sw.js`). When a page requests assets (stylesheets, icons, fonts), the service worker intercepts the request, returning files from the Cache API if available. This satisfies installation requirements.
*   **Micro-Frontend State Machine**: The frontend controller (`app.js`) implements a clear UI state machine (Idle $\to$ Recording/Uploading $\to$ Processing $\to$ Render Results). State switching updates class tags, triggering GPU-accelerated CSS animations.
*   **Dynamic Audio Buffering**: The record panel utilizes the browser's `MediaRecorder` API. It records mono audio signals from the microphone, compiles PCM chunks, encodes them as a WAV blob, and encapsulates them in a standard HTML `File` object for POST uploads.
