# MultiDLFormer

**MultiDLFormer** is a deep learning model based on the **Vision Transformer (ViT)** architecture designed for detecting **deep low-frequency earthquakes (DLFs)**. It takes multichannel waveform data from multiple seismic stations as input and performs **three-class classification**:  
- **DLF**: Deep low-frequency earthquakes  
- **EQ**: Regular earthquakes  
- **Noise**: Non-seismic waveform segments  

---

## 🔍 Model Focus

This model introduces a **multistation approach** to seismic event classification, leveraging both **local and global waveform features** using self-attention. Its architecture is optimized to learn **spatiotemporal patterns** in seismic data that are critical for identifying weak signals like DLFs.

---

## 🧱 Model Architecture

The MultiDLFormer model consists of the following components:

### 1. **Input Tensor**
Shape: `(batch_size, channels=3, stations=10, time=6000)`  
- 3 components: UD, NS, EW  
- 10 stations per input sample  
- 60-second waveform (100 Hz sampling rate)

---

### 2. **Temporal Feature Extraction (Conv2D)**
Applies 2D convolution with:
- **Kernel size**: (1, 50) → temporal dimension
- **Stride**: (1, 25)
- Output: Embedding of **64 dimensions**

This step enhances local temporal features before transformer processing.

---

### 3. **Sequence Formation**
The convolutional output is reshaped into a sequence of tokens:
- One **class token** is prepended
- **Positional encoding** (learnable) is added

---

### 4. **Transformer Encoder (×7 layers)**

Each layer contains:
- **Multi-Head Self-Attention (MHSA)**  
  - 4 heads (each processing 16-dim projections)  
  - Uses **depthwise separable convolution (DSC)** to form query, key, and value  
  - Stride=4 in key/value for efficient computation  
- **Mix-Feedforward Network (Mix-FNN)**  
  - Convolutional MLP with GeLU  
  - Enhances local and nonlinear representations  
- **Standard FFN**, LayerNorm, Residual connections  

---

### 5. **Classification Head**
- Output is extracted from the **class token**
- Fully Connected Layer → **Softmax** over 3 classes  
- The predicted class is the one with **maximum probability**

---

## 📥 Input Configuration

Each training sample consists of:
- 3-component waveform from **10 seismic stations**
- Data augmentation through:
  - Random station selection (with replacement)
  - Reordering (by epicentral distance and random)

---

## 🧪 Loss Function

The model uses **categorical cross-entropy** over softmax outputs.

---

## ⚙️ Training Details

- **Optimizer**: RAdam  
- **Batch Size**: 32  
- **Learning Rate**: 0.001  
- **Early Stopping**: Patience = 20 epochs  
- **Total Samples (after augmentation)**: 688,694 (train), 13,185 (val/test)

---

## 📤 Output

Model returns a 3-class probability vector for each input:

[EQ: 0.11, DLF: 0.87,　Noise: 0.02]  → predicted class: DLF

---

## 📚 Citation
If you use this model, please cite:
Katoh, S., Nagao, H., & Iio, Y. (2025).
MultiDLFormer: A Vision Transformer-Based Deep Learning Model for Detecting Deep Low-Frequency Earthquakes Across Multiple Stations.
Seismological Research Letters (Submitted).
