"""The program records three 6-second voice samples at 44.1 kHz.
Voice 1 is used for Parts A, B and C; all three are compared in Part C.5."""

from math import gcd
from pathlib import Path
import csv
import warnings

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig
import matplotlib.pyplot as plt
import sounddevice as sd


class AudioDSPAssignment:
    def __init__(self, duration=6.0, sample_rate=44100, output_dir="audio_dsp_outputs"):
        self.duration = float(duration)
        self.fs = int(sample_rate)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.signal = None
        self.time = None
        self.n = 0
        self.ts = 1.0 / self.fs
        self.resampled = {}
        self.reconstructed = {}
        self.reconstruction_metrics = {}

    # Utility functions 
    def path(self, filename):
        return self.output_dir / filename

    @staticmethod
    def to_mono_float(audio):
        """Convert mono/stereo integer or floating audio to finite mono float64."""
        audio = np.asarray(audio)
        if audio.ndim == 2:
            audio = np.mean(audio.astype(np.float64), axis=1)
        elif audio.ndim != 1:
            raise ValueError("Audio must be a mono or stereo array.")

        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            scale = float(max(abs(info.min), info.max))
            audio = audio.astype(np.float64) / scale
        else:
            audio = audio.astype(np.float64)

        if audio.size == 0:
            raise ValueError("The audio recording is empty.")
        if not np.all(np.isfinite(audio)):
            raise ValueError("Audio contains NaN or infinite values.")
        return audio

    def save_wav(self, filename, fs, audio):
        """Save normalized floating-point samples as 16-bit PCM WAV."""
        audio = np.asarray(audio, dtype=np.float64)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        wav.write(str(self.path(filename)), int(fs),
                  np.round(audio * 32767).astype(np.int16))

    def set_analysis_signal(self, fs, audio):
        self.fs = int(fs)
        self.signal = self.to_mono_float(audio)
        self.n = len(self.signal)
        self.ts = 1.0 / self.fs
        self.duration = self.n / self.fs
        self.time = np.arange(self.n) / self.fs

    def check_microphone(self):
        """Check whether the selected default input supports the requested rate."""
        try:
            sd.check_input_settings(samplerate=self.fs, channels=1, dtype="float32")
            print(f"Microphone input check passed at {self.fs} Hz.")
        except Exception as exc:
            raise RuntimeError(
                f"The selected microphone does not support {self.fs} Hz "
                f"mono input, or is unavailable. Check your input device/settings. "
                f"Original error: {exc}"
            ) from exc

    # Part A 
    def record_voices(self, count=3):
        recordings = {}
        self.check_microphone()

        for i in range(count):
            input(
                f"\nPress Enter to record Voice {i + 1} "
                f"({self.duration:.1f} seconds)..."
            )
            print("Speak clearly at a normal distance from the microphone.")
            data = sd.rec(
                int(round(self.duration * self.fs)),
                samplerate=self.fs,
                channels=1,
                dtype="float32"
            )
            sd.wait()
            audio = self.to_mono_float(data.flatten())

            # Warn if recording is almost silent or clips at full scale.
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if peak < 1e-4 or rms < 1e-5:
                warnings.warn(f"Voice {i + 1} is nearly silent. Consider recording again.")
            if peak >= 0.999:
                warnings.warn(f"Voice {i + 1} may be clipped. Reduce microphone gain.")

            label = f"Voice {i + 1}"
            recordings[label] = (self.fs, audio)
            self.save_wav(f"voice_recording_{i + 1}.wav", self.fs, audio)
            print(f"{label} saved. Samples: {len(audio)}, peak: {peak:.4f}, RMS: {rms:.4f}")

        # Explicitly preserve the 44.1 kHz reference recording.
        self.save_wav("voice_reference_44100Hz.wav", self.fs, recordings["Voice 1"][1])
        return recordings

    def time_domain_metrics(self):
        if self.signal is None:
            raise RuntimeError("Set an analysis signal first.")

        x = self.signal
        peak = float(np.max(np.abs(x)))
        mean_square = float(np.mean(x ** 2))
        rms = float(np.sqrt(mean_square))
        energy = float(np.sum(x ** 2) / self.fs)

        return {
            "Sampling frequency (Fs)": f"{self.fs} Hz",
            "Number of samples (N)": str(self.n),
            "Signal duration": f"{self.duration:.6f} s",
            "Sampling interval (Ts)": f"{self.ts:.10f} s",
            "Maximum amplitude": f"{np.max(x):.8f}",
            "Minimum amplitude": f"{np.min(x):.8f}",
            "Peak-to-peak amplitude": f"{np.ptp(x):.8f}",
            "Mean-square value / average power": f"{mean_square:.10f}",
            "RMS amplitude": f"{rms:.10f}",
            "Approximate signal energy": f"{energy:.10f} amplitude^2 s",
            "Discrete-time sum of squares": f"{np.sum(x ** 2):.10f}",
            "Peak absolute amplitude": f"{peak:.8f}",
        }

    def save_metrics_text(self, metrics, filename="experimental_results.txt"):
        with open(self.path(filename), "w", encoding="utf-8") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

    def plot_time_domain(self):
        if self.signal is None:
            raise RuntimeError("Set an analysis signal first.")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.time, self.signal, linewidth=0.7,
                label="Recorded voice waveform")
        count = min(250, self.n)
        markerline, stemlines, baseline = ax.stem(
            self.time[:count], self.signal[:count],
            linefmt="r:", markerfmt="ro", basefmt="k-",
            label="Discrete-time samples (zoom)"
        )
        plt.setp(stemlines, linewidth=0.6)
        plt.setp(markerline, markersize=3)
        ax.set_title("Part A: Recorded Voice Signal in Time Domain")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Normalized amplitude")
        ax.grid(True, linestyle=":")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.path("part_a_time_domain.png"), dpi=300)
        plt.close(fig)

    # Part B
    def resample_audio(self, target_rates=(8000, 16000, 44100)):
        """Polyphase resampling includes an anti-aliasing low-pass filter."""
        if self.signal is None:
            raise RuntimeError("Set an analysis signal first.")

        self.resampled = {}
        for rate in target_rates:
            rate = int(rate)
            if rate == self.fs:
                y = self.signal.copy()
            else:
                common = gcd(self.fs, rate)
                y = sig.resample_poly(
                    self.signal, rate // common, self.fs // common
                )
            self.resampled[rate] = y
            self.save_wav(f"voice_resampled_{rate}Hz.wav", rate, y)
            print(f"{self.fs} Hz -> {rate} Hz: {len(y)} samples")
        return self.resampled

    @staticmethod
    def compare_metrics(reference, test):
        """Compare aligned signals; both must represent the same time interval."""
        length = min(len(reference), len(test))
        a = np.asarray(reference[:length], dtype=float)
        b = np.asarray(test[:length], dtype=float)
        error = a - b
        mse = float(np.mean(error ** 2))
        rmse = float(np.sqrt(mse))
        signal_power = float(np.mean(a ** 2))
        if mse <= np.finfo(float).tiny:
            snr = float("inf")
        elif signal_power <= np.finfo(float).tiny:
            snr = float("-inf")
        else:
            snr = float(10 * np.log10(signal_power / mse))

        if np.std(a) < 1e-15 or np.std(b) < 1e-15:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(a, b)[0, 1])

        return {"MSE": mse, "RMSE": rmse, "SNR_dB": snr,
                "Correlation": corr, "Compared_samples": length}

    def compare_reconstruction(self):
        """Downsample then reconstruct to original Fs and compare quantitatively."""
        if not self.resampled:
            raise RuntimeError("Run resample_audio() first.")

        rates = [r for r in self.resampled if r != self.fs]
        if not rates:
            raise RuntimeError("No lower sampling rates available for reconstruction.")

        fig, axes = plt.subplots(len(rates), 1,
                                 figsize=(12, 4 * len(rates)), squeeze=False)
        self.reconstructed = {}
        self.reconstruction_metrics = {}

        segment_duration = min(0.03, self.duration)
        count = max(2, int(segment_duration * self.fs))
        t_ms = self.time[:count] * 1000

        for ax, rate in zip(axes[:, 0], rates):
            common = gcd(rate, self.fs)
            y = sig.resample_poly(
                self.resampled[rate], self.fs // common, rate // common
            )

            # Match original length without changing the aligned portion.
            if len(y) < self.n:
                y = np.pad(y, (0, self.n - len(y)))
            else:
                y = y[:self.n]

            self.reconstructed[rate] = y
            metrics = self.compare_metrics(self.signal, y)
            self.reconstruction_metrics[rate] = metrics

            self.save_wav(f"voice_reconstructed_from_{rate}Hz.wav", self.fs, y)
            ax.plot(t_ms, self.signal[:count], label="Original 44.1 kHz")
            ax.plot(t_ms, y[:count], "--", label=f"Reconstructed from {rate} Hz")
            ax.set_title(f"Part B: Original vs reconstructed ({rate} Hz)")
            ax.set_xlabel("Time (milliseconds)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, linestyle=":")
            ax.legend()

            print(
                f"{rate} Hz round trip: MSE={metrics['MSE']:.6e}, "
                f"RMSE={metrics['RMSE']:.6e}, SNR={metrics['SNR_dB']:.2f} dB, "
                f"correlation={metrics['Correlation']:.6f}"
            )

        fig.tight_layout()
        fig.savefig(self.path("part_b_resampling_comparison.png"), dpi=300)
        plt.close(fig)

        with open(self.path("part_b_quality_metrics.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Sampling rate (Hz)", "MSE", "RMSE", "SNR (dB)",
                             "Correlation", "Compared samples"])
            for rate, m in self.reconstruction_metrics.items():
                writer.writerow([rate, m["MSE"], m["RMSE"], m["SNR_dB"],
                                 m["Correlation"], m["Compared_samples"]])

    def windowed_sinc_reconstruction(self, low_rate_signal, low_fs=8000):
        """
        Finite, Kaiser-windowed sinc interpolation on a short display interval.
        This is an approximation to ideal band-limited reconstruction.
        """
        start = min(1.0, max(0.0, self.duration * 0.25))
        end = min(start + 0.02, self.duration)
        if end <= start:
            raise ValueError("Recording is too short for sinc reconstruction.")

        low = np.asarray(low_rate_signal, dtype=float)
        t_low_all = np.arange(len(low)) / low_fs
        half_width = 16  # interpolation kernel half-width in low-rate samples
        t_grid = np.linspace(start, end, 2000)
        reconstructed = np.zeros_like(t_grid)

        for i, t in enumerate(t_grid):
            distance_samples = (t - t_low_all) * low_fs
            mask = np.abs(distance_samples) <= half_width
            d = distance_samples[mask]
            # Kaiser window is zero at the finite kernel boundary.
            window = np.kaiser(2 * half_width + 1, beta=8.6)
            indices = np.clip(
                np.round(d + half_width).astype(int), 0, 2 * half_width
            )
            weights = np.sinc(d) * window[indices]
            weight_sum = np.sum(weights)
            if abs(weight_sum) > 1e-12:
                reconstructed[i] = np.sum(low[mask] * weights) / weight_sum

        original_mask = (self.time >= start) & (self.time <= end)
        shown_mask = (t_low_all >= start) & (t_low_all <= end)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot((self.time[original_mask] - start) * 1000,
                self.signal[original_mask], label="Original 44.1 kHz samples")
        ax.stem((t_low_all[shown_mask] - start) * 1000,
                low[shown_mask], linefmt="r:", markerfmt="ro",
                basefmt="k-", label="8 kHz samples")
        ax.plot((t_grid - start) * 1000, reconstructed, "b--",
                linewidth=1.2, label="Kaiser-windowed sinc interpolation")
        ax.set_title("Part B: Windowed-Sinc Reconstruction (20 ms)")
        ax.set_xlabel("Time from selected interval (milliseconds)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle=":")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.path("part_b_sinc_reconstruction.png"), dpi=300)
        plt.close(fig)

    def aliasing_demo(self):
        """Illustrate 1500 Hz sampled at 2000 Hz: alias frequency is 500 Hz."""
        tone = 1500.0
        fs_under = 2000.0
        t = np.linspace(0, 0.004, 1500)
        original = np.sin(2 * np.pi * tone * t)

        sample_t = np.arange(int(np.floor(0.004 * fs_under)) + 1) / fs_under
        sample_x = np.sin(2 * np.pi * tone * sample_t)

        alias_signed = ((tone + fs_under / 2) % fs_under) - fs_under / 2
        alias = np.sin(2 * np.pi * alias_signed * t)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(t, original, label="Original 1500 Hz tone")
        ax.stem(sample_t, sample_x, linefmt="m:", markerfmt="mo",
                basefmt="k-", label="Samples at 2000 Hz")
        ax.plot(t, alias, "r--", label=f"Aliased tone ({abs(alias_signed):.0f} Hz)")
        ax.set_title("Part B: Undersampling and Aliasing Demonstration")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle=":")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.path("part_b_aliasing_demo.png"), dpi=300)
        plt.close(fig)
        print(
            f"Nyquist frequency = {fs_under/2:.0f} Hz. "
            f"{tone:.0f} Hz sampled at {fs_under:.0f} Hz appears at "
            f"{abs(alias_signed):.0f} Hz (alias frequency)."
        )

    # Part C 
    def fourier_analysis(self, max_peaks=8):
        if self.signal is None:
            raise RuntimeError("Set an analysis signal first.")

        window = sig.windows.hann(self.n, sym=False)
        spectrum = np.fft.rfft(self.signal * window)
        frequencies = np.fft.rfftfreq(self.n, d=1 / self.fs)

        magnitude = np.abs(spectrum) / np.sum(window)
        if self.n % 2 == 0:
            magnitude[1:-1] *= 2
        else:
            magnitude[1:] *= 2
        phase = np.unwrap(np.angle(spectrum))

        # Search only in a useful voice-frequency range; omit DC.
        upper = min(4000.0, self.fs / 2)
        mask = (frequencies >= 60) & (frequencies <= upper)
        f_voice = frequencies[mask]
        m_voice = magnitude[mask]

        peaks, properties = sig.find_peaks(
            m_voice, prominence=max(np.max(m_voice) * 0.01, 1e-8),
            distance=max(1, int(20 / (self.fs / self.n)))
        )
        if len(peaks):
            order = np.argsort(m_voice[peaks])[::-1][:max_peaks]
            selected = peaks[order]
            peak_rows = [(float(f_voice[p]), float(m_voice[p])) for p in selected]
        else:
            peak_rows = []

        print("\nMajor spectral peaks (prominent components, not necessarily pitch):")
        for index, (frequency, amplitude) in enumerate(peak_rows, 1):
            print(f"{index}. {frequency:.2f} Hz, amplitude {amplitude:.6f}")
        if not peak_rows:
            print("No prominent peaks detected in the selected voice range.")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
        ax1.plot(frequencies, magnitude, color="crimson", linewidth=0.8)
        ax1.set_title("Part C: Single-Sided FFT Magnitude Spectrum")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(0, min(5000, self.fs / 2))
        ax1.grid(True, linestyle=":")
        ax2.plot(frequencies, phase, color="purple", linewidth=0.6)
        ax2.set_title("Part C: FFT Phase Spectrum")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (radians)")
        ax2.set_xlim(0, min(5000, self.fs / 2))
        ax2.grid(True, linestyle=":")
        fig.tight_layout()
        fig.savefig(self.path("part_c_fourier_spectra.png"), dpi=300)
        plt.close(fig)

        with open(self.path("part_c_major_frequency_peaks.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Rank", "Frequency (Hz)", "Magnitude"])
            for rank, (frequency, amplitude) in enumerate(peak_rows, 1):
                writer.writerow([rank, frequency, amplitude])

        return frequencies, magnitude, phase, peak_rows

    def resolution_analysis(self):
        """Compare real record lengths and report delta-f = Fs/N."""
        if self.signal is None:
            raise RuntimeError("Set an analysis signal first.")

        candidates = [512, 4096, self.n]
        sizes = sorted(set(min(int(k), self.n) for k in candidates if k >= 2))
        fig, axes = plt.subplots(len(sizes), 1,
                                 figsize=(12, 3.5 * len(sizes)), squeeze=False)

        print("\nPart C.4: Frequency resolution")
        rows = []
        for ax, size in zip(axes[:, 0], sizes):
            segment = self.signal[:size]
            win = sig.windows.hann(size, sym=False)
            spec = np.fft.rfft(segment * win)
            freq = np.fft.rfftfreq(size, d=1 / self.fs)
            amp = np.abs(spec) / np.sum(win)
            if size % 2 == 0:
                amp[1:-1] *= 2
            else:
                amp[1:] *= 2

            delta_f = self.fs / size
            rows.append((size, size / self.fs, delta_f))
            print(f"N={size}, duration={size/self.fs:.6f} s, "
                  f"frequency resolution={delta_f:.4f} Hz/bin")
            ax.plot(freq, amp, linewidth=0.8)
            ax.set_title(f"N={size}, frequency resolution={delta_f:.2f} Hz")
            ax.set_xlim(0, min(4000, self.fs / 2))
            ax.set_ylabel("Amplitude")
            ax.grid(True, linestyle=":")

        axes[-1, 0].set_xlabel("Frequency (Hz)")
        fig.suptitle("Part C.4: FFT Resolution for Different Sample Counts")
        fig.tight_layout()
        fig.savefig(self.path("part_c4_frequency_resolution.png"), dpi=300)
        plt.close(fig)

        with open(self.path("part_c4_resolution.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["N", "Record duration (s)", "Resolution (Hz/bin)"])
            writer.writerows(rows)

    def compare_voices(self, recordings):
        fig, ax = plt.subplots(figsize=(12, 6))
        results = []

        for label, (fs, audio) in recordings.items():
            x = self.to_mono_float(audio)
            if len(x) < 2:
                print(f"Skipping {label}: recording too short.")
                continue

            nperseg = min(4096, len(x))
            freq, psd = sig.welch(
                x, fs=fs, window="hann", nperseg=nperseg,
                noverlap=nperseg // 2, scaling="density"
            )
            mask = (freq >= 50) & (freq <= min(5000, fs / 2))
            if not np.any(mask):
                continue

            psd_db = 10 * np.log10(psd + 1e-15)
            psd_db -= np.max(psd_db[mask])
            ax.plot(freq[mask], psd_db[mask], linewidth=1.3, label=label)

            local_freq = freq[mask]
            local_psd = psd[mask]
            dominant = float(local_freq[np.argmax(local_psd)])
            results.append((label, dominant, float(np.max(x)), float(np.sqrt(np.mean(x**2)))))
            print(f"{label}: strongest PSD component = {dominant:.2f} Hz")

        ax.set_title("Part C.5: Comparison of Voice Frequency Responses")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalized PSD (dB relative to each recording peak)")
        ax.set_xlim(50, 5000)
        ax.set_ylim(-60, 5)
        ax.grid(True, linestyle=":")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.path("part_c5_voice_frequency_comparison.png"), dpi=300)
        plt.close(fig)

        with open(self.path("part_c5_voice_comparison.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Voice", "Dominant PSD frequency (Hz)",
                             "Maximum amplitude", "RMS amplitude"])
            writer.writerows(results)

    # Playback and report outputs 
    def playback(self):
        print("\nAudio playback comparison:")
        print("Playing original recording...")
        sd.play(self.signal, self.fs)
        sd.wait()

        for rate, low in self.resampled.items():
            if rate == self.fs:
                continue
            print(f"Playing resampled audio at {rate} Hz...")
            sd.play(low, rate)
            sd.wait()
            if rate in self.reconstructed:
                print(f"Playing reconstruction from {rate} Hz at {self.fs} Hz...")
                sd.play(self.reconstructed[rate], self.fs)
                sd.wait()

    def write_summary(self, time_metrics, peaks):
        with open(self.path("experimental_results.txt"), "w", encoding="utf-8") as f:
            f.write("PART A: TIME-DOMAIN MEASUREMENTS\n")
            for key, value in time_metrics.items():
                f.write(f"{key}: {value}\n")

            f.write("\nPART B: RECONSTRUCTION QUALITY\n")
            for rate, metrics in self.reconstruction_metrics.items():
                f.write(f"\nFrom {rate} Hz:\n")
                for key, value in metrics.items():
                    f.write(f"  {key}: {value}\n")

            f.write("\nPART C: PROMINENT FFT PEAKS\n")
            for rank, (frequency, amplitude) in enumerate(peaks, 1):
                f.write(f"{rank}. {frequency:.3f} Hz, magnitude={amplitude:.8f}\n")

        print(f"\nResults and graphs saved in: {self.output_dir.resolve()}")


def main():
    pipeline = AudioDSPAssignment(
        duration=6.0,
        sample_rate=44100,
        output_dir="audio_dsp_outputs"
    )

    print("=== TEZPUR UNIVERSITY: AUDIO DSP ASSIGNMENT ===")
    print("Record three distinct voice samples, preferably from three speakers.")
    recordings = pipeline.record_voices(count=3)

    # Part C.5: compare all three recordings.
    pipeline.compare_voices(recordings)

    # Use Voice 1 consistently for Parts A, B and C.
    fs, voice1 = recordings["Voice 1"]
    pipeline.set_analysis_signal(fs, voice1)

    # Part A
    print("\n=== PART A: TIME-DOMAIN ANALYSIS ===")
    time_metrics = pipeline.time_domain_metrics()
    for key, value in time_metrics.items():
        print(f"{key}: {value}")
    pipeline.save_metrics_text(time_metrics)
    pipeline.plot_time_domain()

    # Part B
    print("\n=== PART B: SAMPLING AND RECONSTRUCTION ===")
    resampled = pipeline.resample_audio(target_rates=(8000, 16000, 44100))
    pipeline.compare_reconstruction()
    pipeline.windowed_sinc_reconstruction(resampled[8000], low_fs=8000)
    pipeline.aliasing_demo()

    # Part C
    print("\n=== PART C: FOURIER ANALYSIS ===")
    _, _, _, peaks = pipeline.fourier_analysis(max_peaks=8)
    pipeline.resolution_analysis()

    pipeline.write_summary(time_metrics, peaks)

    choice = input(
        "\nPlay original, resampled and reconstructed audio? (y/n): "
    ).strip().lower()
    if choice == "y":
        pipeline.playback()
    else:
        print("Audio playback skipped.")

    print("\n=== ASSIGNMENT PROCESSING COMPLETED ===")
    print("Attach the seven PNG graphs to your handwritten report.")
    print("Use the CSV/TXT outputs for actual numerical observations.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as exc:
        print(f"\nAn error occurred: {exc}")
        raise

