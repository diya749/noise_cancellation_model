# 🎙️ Audio Noise Cancellation — Deep Learning with U-Net

A deep learning model that removes background noise from audio recordings using a U-Net architecture with spectral masking. Built from scratch as a learning project by someone with no prior coding background, using Python, TensorFlow, and Google Colab.

---

## 🔍 How It Works

The model follows a spectrogram-based approach:

1. **Audio → Spectrogram** — converts raw audio into a 2D frequency-time image using Short-Time Fourier Transform (STFT)
2. **U-Net Mask Prediction** — the model predicts a mask (values 0–1) for every frequency bin at every time step
3. **Mask × Spectrogram** — the mask suppresses noise frequencies while preserving voice frequencies
4. **Spectrogram → Audio** — reconstructs cleaned audio using inverse STFT with correct phase reconstruction
5. **Overlap-Add** — long audio is processed in 2-second chunks with 0.25s overlap for seamless reconstruction

---

## 🧠 Model Architecture

- **Type:** U-Net with skip connections
- **Input:** Padded spectrogram (264 × 248 × 1)
- **Encoder:** Conv2D layers with 32 → 64 → 128 → 256 filters + MaxPooling
- **Bottleneck:** 256 filters — most abstract compressed representation
- **Decoder:** UpSampling + skip connections + Conv2D layers (128 → 64 → 32)
- **Output:** Sigmoid mask multiplied against input spectrogram
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam

---

## ⚙️ Key Parameters

| Parameter | Value | Purpose |
|---|---|---|
| Sample Rate | 16,000 Hz | Standard for voice quality |
| N_FFT | 512 | Frequency resolution of STFT |
| Hop Length | 128 | Time resolution (75% window overlap) |
| Chunk Duration | 2 seconds | Model input length |
| Overlap Duration | 0.25 seconds | Smooth chunk rejoining |
| Learning Rate | 0.001 | Adam optimizer starting LR |
| Batch Size | 4 | Colab memory constraint |
| Validation Split | 20% | Generalization measurement |

---

## 📁 Project Structure

```
├── model.ipynb          # Main notebook — training + inference
├── weights.weights.h5    # Trained model weights
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/audio-noise-cancellation.git
```

### 2. Install dependencies

### 3. Open in Google Colab
Upload `Untitled2.ipynb` to [Google Colab](https://colab.research.google.com) for best results — GPU acceleration significantly speeds up training and inference.

### 4. For inference only (no training needed)
- Upload `best_model.weights.h5` to your Colab session
- Set `USER_AUDIO_PATH` to your noisy audio file path
- Run the inference cells only
---

## 📋 Requirements

```
tensorflow>=2.10
librosa>=0.9
soundfile>=0.11
numpy>=1.21
matplotlib>=3.5
pydub>=0.25
```

---

## 🔧 Features

- ✅ Supports any audio length via chunking
- ✅ Accepts .wav, .mp3, .m4a, .mp4 input formats
- ✅ Automatic sample rate resampling
- ✅ Global RMS normalization — output volume matches input
- ✅ Overlap-add reconstruction — no glitches at chunk boundaries
- ✅ Correct polar-form phase reconstruction
- ✅ Batch inference — process multiple files at once

---

## 📈 Training Results

The model shows consistent convergence with training and validation loss tracking closely, indicating good generalization rather than overfitting.

Training loss and validation loss both converge around **0.13–0.15 MSE** on the Kaggle dataset.

---

## 🛣️ Planned Improvements

- [ ] Fine-tune on personal voice recordings for better personalization
- [ ] Add Gradio web interface for easy public use on Hugging Face Spaces
- [ ] Experiment with combined MSE + Spectral Convergence loss
- [ ] Add log-scale magnitude compression for better handling of quiet sounds
- [ ] Expand dataset with diverse noise types

---

## 💡 What I Learned

This project was built entirely as a learning exercise with no prior coding background. Key concepts learned through building:

- Audio signal processing fundamentals (FFT, STFT, spectrograms, phase reconstruction)
- U-Net architecture and why skip connections matter
- Spectral masking vs direct audio reconstruction
- Overlap-add reconstruction for seamless audio chunking
- Training dynamics — overfitting, validation loss, early stopping, learning rate scheduling
- The difference between DSP-based and ML-based noise cancellation approaches

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Author

Built by Diya as a self-directed ML learning project.
(the project is going on, updates are made accordingly on Github
