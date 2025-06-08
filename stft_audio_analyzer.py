import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal
import os

def apply_audio_filters(samples, sample_rate, apply_lowpass=True):
    """Apply essential audio filters for clean analysis"""
    
    filtered_samples = samples.copy()
    filter_description = []
    
    # 1. High-pass filter to remove sub-sonic noise and DC offset (ESSENTIAL)
    # Remove frequencies below 20 Hz
    sos_hp = scipy.signal.butter(4, 20, btype='high', fs=sample_rate, output='sos')
    filtered_samples = scipy.signal.sosfilt(sos_hp, filtered_samples)
    filter_description.append("High-pass (20 Hz)")
    
    # 2. Optional low-pass filter around 15 kHz
    if apply_lowpass:
        cutoff = 15000  # 15 kHz cutoff
        if cutoff < sample_rate // 2:  # Only apply if below Nyquist
            sos_lp = scipy.signal.butter(6, cutoff, btype='low', fs=sample_rate, output='sos')
            filtered_samples = scipy.signal.sosfilt(sos_lp, filtered_samples)
            filter_description.append(f"Low-pass ({cutoff} Hz)")
    
    return filtered_samples, " → ".join(filter_description)

def analyze_audio_stft_comparison(file_paths=["My Way.mp3", "my way cover.mp3"], target_sample_rate=40000, window_size=4096, apply_lowpass=True):
    """STFT-based audio analyzer that compares multiple audio files"""
    
    print("=== STFT Audio Comparison Analyzer ===\n")
    
    # Store data for both files
    audio_data = []
    labels = []
    
    for i, file_path in enumerate(file_paths):
        print(f"Processing file {i+1}/2: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found!")
            continue
        
        try:
            # Load and immediately convert to mono and downsample
            print("  Loading audio... (converting to mono and resampling)")
            audio, original_sample_rate = librosa.load(
                file_path, 
                sr=target_sample_rate,  # Downsample during loading for efficiency
                mono=True               # Convert to mono during loading
            )
            print(f"  ✓ Audio loaded and processed successfully!")
            print(f"    - Duration: {len(audio) / target_sample_rate:.2f} seconds")
            print(f"    - Total samples: {len(audio)}")
            
            # Apply essential filters
            print(f"  Applying filters...")
            samples, filter_info = apply_audio_filters(audio, target_sample_rate, apply_lowpass)
            print(f"  ✓ Filtering completed: {filter_info}")
            
            # Store the processed audio
            audio_data.append(samples)
            
            # Create a label for the file
            if "cover" in file_path.lower():
                labels.append("Cover Version")
            else:
                labels.append("Original")
                
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue
        
        print()
    
    if len(audio_data) < 2:
        print("Error: Need at least 2 valid audio files for comparison!")
        return
    
    # Trim longer audio to match shorter one for fair comparison
    print(f"Synchronizing audio lengths for fair comparison...")
    
    durations = [len(samples) / target_sample_rate for samples in audio_data]
    min_duration = min(durations)
    min_samples = int(min_duration * target_sample_rate)
    
    print(f"  - Original durations: {durations[0]:.2f}s, {durations[1]:.2f}s")
    print(f"  - Trimming both to: {min_duration:.2f}s ({min_samples} samples)")
    
    # Trim all audio to the same length
    for i in range(len(audio_data)):
        if len(audio_data[i]) > min_samples:
            audio_data[i] = audio_data[i][:min_samples]
            print(f"  ✓ {labels[i]} trimmed from {durations[i]:.2f}s to {min_duration:.2f}s")
        else:
            print(f"  ✓ {labels[i]} already {durations[i]:.2f}s (no trimming needed)")
    
    print()
    
    print(f"Performing STFT analysis on both files...")
    
    # Process both audio files with STFT
    stft_data = []
    hop_length = window_size // 4  # 75% overlap for good time resolution
    
    print(f"  - Window size: {window_size} samples ({window_size/target_sample_rate*1000:.1f} ms)")
    print(f"  - Hop length: {hop_length} samples ({hop_length/target_sample_rate*1000:.1f} ms)")
    print(f"  - Overlap: {(1 - hop_length/window_size)*100:.1f}%")
    
    for i, (samples, label) in enumerate(zip(audio_data, labels)):
        try:
            print(f"  Processing STFT for {label}...")
            
            # Perform STFT using librosa
            stft_result = librosa.stft(
                samples, 
                n_fft=window_size, 
                hop_length=hop_length,
                window='hann'
            )
            
            # Get magnitude spectrogram in dB
            magnitude_spectrogram = np.abs(stft_result)
            magnitude_db = librosa.amplitude_to_db(magnitude_spectrogram, ref=np.max)
            
            # Create frequency and time axes
            frequencies = librosa.fft_frequencies(sr=target_sample_rate, n_fft=window_size)
            times = librosa.frames_to_time(
                np.arange(magnitude_spectrogram.shape[1]), 
                sr=target_sample_rate, 
                hop_length=hop_length
            )
            
            # Store the data
            stft_data.append({
                'magnitude_db': magnitude_db,
                'frequencies': frequencies,
                'times': times,
                'label': label,
                'samples': samples
            })
            
            print(f"  ✓ {label} STFT completed! ({len(frequencies)} freq bins, {len(times)} time frames)")
            
        except Exception as e:
            print(f"  Error performing STFT on {label}: {e}")
            return
    
    print(f"✓ All STFT analysis completed!")
    
    print(f"\nCreating comparison spectrograms...")
    
    try:
        # Create the comparison plot - 3 rows, 2 columns
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        
        # Get common frequency range for both files
        frequencies = stft_data[0]['frequencies']
        audible_mask = (frequencies >= 20) & (frequencies <= 20000)
        audible_frequencies = frequencies[audible_mask]
        
        colors = ['blue', 'red']
        
        # Row 1: Full audible range spectrograms side by side
        for i, data in enumerate(stft_data):
            audible_magnitude_db = data['magnitude_db'][audible_mask, :]
            
            im = axes[0, i].imshow(
                audible_magnitude_db, 
                aspect='auto', 
                origin='lower',
                extent=[data['times'][0], data['times'][-1], audible_frequencies[0], audible_frequencies[-1]],
                cmap='viridis',
                vmin=-80, vmax=0
            )
            
            # Add peak frequency tracking over time
            peak_freqs = []
            times_for_peaks = []
            
            # Find dominant frequency at each time frame (every 10th frame for clarity)
            step = max(1, len(data['times']) // 200)  # Limit to ~200 points for visibility
            for t_idx in range(0, len(data['times']), step):
                # Find peak in audible range for this time frame
                time_slice = audible_magnitude_db[:, t_idx]
                peak_idx = np.argmax(time_slice)
                peak_freq = audible_frequencies[peak_idx]
                
                # Only show if peak is significant (above -60 dB)
                if time_slice[peak_idx] > -60:
                    peak_freqs.append(peak_freq)
                    times_for_peaks.append(data['times'][t_idx])
            
            # Plot peak frequency line
            if len(peak_freqs) > 0:
                axes[0, i].plot(times_for_peaks, peak_freqs, 'r-', linewidth=2, alpha=0.8, 
                               label='Peak Frequency')
                axes[0, i].scatter(times_for_peaks, peak_freqs, c='red', s=8, alpha=0.7, 
                                  zorder=5)
            
            axes[0, i].set_title(f'{data["label"]} - Full Audible Range (20 Hz - 20 kHz)', fontsize=10)
            axes[0, i].set_xlabel('Time (seconds)')
            axes[0, i].set_ylabel('Frequency (Hz)')
            axes[0, i].set_yscale('log')
            axes[0, i].set_ylim(20, 20000)
            axes[0, i].legend(loc='upper right', fontsize=8)
            plt.colorbar(im, ax=axes[0, i], label='Magnitude (dB)')
        
        # Row 2: Average frequency spectrum comparison
        axes[1, 0].set_title('Average Frequency Spectrum Comparison', fontsize=10)
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Magnitude (dB)')
        axes[1, 0].set_xscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(20, 20000)
        
        for i, data in enumerate(stft_data):
            avg_spectrum = np.mean(np.abs(librosa.stft(data['samples'], n_fft=window_size, hop_length=hop_length)), axis=1)
            audible_avg_spectrum = avg_spectrum[audible_mask]
            
            axes[1, 0].semilogx(
                audible_frequencies, 
                librosa.amplitude_to_db(audible_avg_spectrum, ref=np.max), 
                color=colors[i], 
                label=data['label'],
                linewidth=2
            )
        
        axes[1, 0].legend()
        
        # Row 2, Column 2: Frequency difference plot
        axes[1, 1].set_title('Frequency Spectrum Difference (Original - Cover)', fontsize=10)
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Magnitude Difference (dB)')
        axes[1, 1].set_xscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(20, 20000)
        
        # Calculate difference between original and cover
        orig_spectrum = np.mean(np.abs(librosa.stft(stft_data[0]['samples'], n_fft=window_size, hop_length=hop_length)), axis=1)
        cover_spectrum = np.mean(np.abs(librosa.stft(stft_data[1]['samples'], n_fft=window_size, hop_length=hop_length)), axis=1)
        
        orig_audible = librosa.amplitude_to_db(orig_spectrum[audible_mask], ref=np.max)
        cover_audible = librosa.amplitude_to_db(cover_spectrum[audible_mask], ref=np.max)
        difference = orig_audible - cover_audible
        
        axes[1, 1].semilogx(audible_frequencies, difference, color='green', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Row 3: Low frequency detailed comparison (20 Hz - 2 kHz)
        low_freq_mask = (frequencies >= 20) & (frequencies <= 2000)
        low_frequencies = frequencies[low_freq_mask]
        
        for i, data in enumerate(stft_data):
            low_magnitude_db = data['magnitude_db'][low_freq_mask, :]
            
            im = axes[2, i].imshow(
                low_magnitude_db,
                aspect='auto',
                origin='lower',
                extent=[data['times'][0], data['times'][-1], low_frequencies[0], low_frequencies[-1]],
                cmap='viridis',
                vmin=-80, vmax=0
            )
            
            # Add peak frequency tracking for low frequencies
            low_peak_freqs = []
            low_times_for_peaks = []
            
            # Find dominant frequency in low range at each time frame
            step = max(1, len(data['times']) // 150)  # Fewer points for low freq view
            for t_idx in range(0, len(data['times']), step):
                # Find peak in low frequency range for this time frame
                time_slice = low_magnitude_db[:, t_idx]
                peak_idx = np.argmax(time_slice)
                peak_freq = low_frequencies[peak_idx]
                
                # Only show if peak is significant (above -50 dB for low freq)
                if time_slice[peak_idx] > -50:
                    low_peak_freqs.append(peak_freq)
                    low_times_for_peaks.append(data['times'][t_idx])
            
            # Plot peak frequency line for low frequencies
            if len(low_peak_freqs) > 0:
                axes[2, i].plot(low_times_for_peaks, low_peak_freqs, 'yellow', linewidth=3, 
                               alpha=0.9, label='Dominant Bass/Mid')
                axes[2, i].scatter(low_times_for_peaks, low_peak_freqs, c='orange', s=12, 
                                  alpha=0.8, zorder=5, edgecolors='black', linewidth=0.5)
            
            axes[2, i].set_title(f'{data["label"]} - Music Fundamentals (20 Hz - 2 kHz)', fontsize=10)
            axes[2, i].set_xlabel('Time (seconds)')
            axes[2, i].set_ylabel('Frequency (Hz)')
            axes[2, i].legend(loc='upper right', fontsize=8)
            plt.colorbar(im, ax=axes[2, i], label='Magnitude (dB)')
        
        plt.tight_layout(pad=2.0)  # Add more padding between subplots
        plt.subplots_adjust(hspace=0.35, wspace=0.25)  # Adjust spacing between rows and columns
        
        print(f"✓ Comparison spectrograms created!")
        
        # Save the comparison plot
        plt.savefig('stft_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Comparison spectrograms saved as 'stft_comparison.png'")
        
        # Show the plot
        print("✓ Displaying comparison spectrograms...")
        plt.show()
        
        # Print comparison analysis
        print(f"\n=== Comparison Analysis ===")
        
        for i, data in enumerate(stft_data):
            print(f"\n{data['label']}:")
            
            # Calculate average spectrum for analysis
            avg_spectrum = np.mean(np.abs(librosa.stft(data['samples'], n_fft=window_size, hop_length=hop_length)), axis=1)
            audible_avg = avg_spectrum[audible_mask]
            
            # Find top 5 frequencies
            top_indices = np.argsort(audible_avg)[-5:][::-1]
            
            for j, idx in enumerate(top_indices):
                freq = audible_frequencies[idx]
                magnitude = librosa.amplitude_to_db(audible_avg[idx], ref=np.max)
                
                # Add frequency range description
                if freq < 250:
                    freq_type = "Bass"
                elif freq < 500:
                    freq_type = "Low Mid"
                elif freq < 2000:
                    freq_type = "Mid"
                elif freq < 6000:
                    freq_type = "High Mid"
                else:
                    freq_type = "Treble"
                    
                print(f"  {j+1}. {freq:.1f} Hz ({magnitude:.1f} dB) - {freq_type}")
        
        # Frequency difference analysis
        print(f"\n=== Key Differences ===")
        significant_diff = np.abs(difference) > 3  # Differences > 3dB
        significant_freqs = audible_frequencies[significant_diff]
        significant_diffs = difference[significant_diff]
        
        if len(significant_freqs) > 0:
            print("Significant frequency differences (>3dB):")
            for freq, diff in zip(significant_freqs[:10], significant_diffs[:10]):  # Top 10
                direction = "Original stronger" if diff > 0 else "Cover stronger"
                print(f"  {freq:.1f} Hz: {abs(diff):.1f} dB ({direction})")
        else:
            print("No significant frequency differences found (both versions are very similar)")
        
    except Exception as e:
        print(f"Error creating spectrograms: {e}")
        return
    
    print("\n=== STFT Comparison Analysis Complete! ===")

if __name__ == "__main__":
    # Run the STFT comparison analysis
    print("Analyzing and comparing audio files with STFT (Short-Time Fourier Transform)")
    print("This creates comparison spectrograms showing differences between original and cover")
    print("Includes audio filtering for improved analysis quality\n")
    
    analyze_audio_stft_comparison(
        file_paths=["My Way.mp3", "my way cover.mp3"],  # Original and cover versions
        target_sample_rate=40000,  # 40 kHz sampling rate for human audible range
        window_size=4096,  # Window size for good frequency resolution
        apply_lowpass=True  # Apply low-pass filter at 15 kHz
    ) 