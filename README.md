
# Audio Signal Acquisition, Sampling, Reconstruction and Fourier Analysis

**Department of Electronics and Communication Engineering**  
**Tezpur University | Autumn 2026**

## About the Project

This project implements a Python-based Digital Signal Processing (DSP) system for acquiring, analysing, resampling and reconstructing human voice signals.

It demonstrates fundamental concepts of Signals and Systems through practical audio processing.

## Objectives

- Audio signal acquisition using a microphone.
- Time-domain signal analysis.
- Sampling and resampling at different frequencies.
- Signal reconstruction using interpolation.
- Demonstration of aliasing and the Nyquist theorem.
- Fourier Transform (FFT) analysis.
- Frequency resolution analysis.
- Comparison of different voice recordings.

## Technologies Used

- Python
- NumPy
- SciPy
- Matplotlib
- SoundDevice

## Sampling Frequencies

| Sampling Frequency | Application |
|---|---|
| 8 kHz | Downsampled audio |
| 16 kHz | Downsampled audio |
| 44.1 kHz | Reference audio |

## Project Structure

```text
Audio-DSP-Signals-Systems/
│
├── main.py
├── README.md
├── .gitignore
│
└── audio_dsp_outputs/
    ├── part_a_time_domain.png
    ├── part_b_resampling_comparison.png
    ├── part_b_sinc_reconstruction.png
    ├── part_b_aliasing_demo.png
    ├── part_c_fourier_spectra.png
    ├── part_c4_frequency_resolution.png
    └── part_c5_voice_frequency_comparison.png
```

## Installation

Install the required libraries:

```bash
python -m pip install numpy scipy matplotlib sounddevice
```

## How to Run

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Open the integrated terminal.
4. Run:

```bash
python main.py
```

5. Follow the instructions to record three voice samples.

## Expected Outputs

- Time-domain waveform.
- Original vs reconstructed waveform.
- Windowed-sinc reconstruction.
- Aliasing demonstration.
- FFT magnitude and phase spectra.
- Frequency resolution comparison.
- Voice frequency comparison.

Generated graphs and numerical results are saved in the `audio_dsp_outputs` folder.

## Academic Context

This project was developed as part of the Signals and Systems assignment at Tezpur University, Department of Electronics and Communication Engineering.

**Author:** MOHD SHAJEB ANSARI

**Branch:** Electronics and Communication Engineering (ECE)
