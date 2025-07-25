import numpy as np
import librosa
import matplotlib.pyplot as plt
import csv
import os

folder_path = "/your path/fold_audio"
output_txt_path = "results.txt"
output_csv_path = "results.csv"

n_mels_range = range(10, 46, 5)
hop_length_variants = [None, None, None]
threshold_range = np.arange(0.1, 0.5, 0.1)

audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

with open(output_csv_path, "w", newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    header = ["Audio File", "n_mels", "hop_length", "Threshold", "Groups of Zero Columns"]
    csv_writer.writerow(header)

    with open(output_txt_path, "w") as txt_file:
        for audio_file in audio_files:
            audio_path = os.path.join(folder_path, audio_file)
            y, sr = librosa.load(audio_path)
            window_size = int(0.02 * sr)
            hop_length_variants = [window_size // 4, window_size // 2, (window_size // 4) * 3]

            txt_file.write(f"Processing file: {audio_file}\n")

            for n_mels in n_mels_range:
                for hop_length in hop_length_variants:
                    mel_spectrogram = librosa.feature.melspectrogram(
                        y=y, sr=sr, n_fft=window_size, hop_length=hop_length,
                        n_mels=n_mels, window='hann'
                    )
                    spectrogram = mel_spectrogram

                    spectrogram_matrix = spectrogram.reshape((spectrogram.shape[0], -1))
                    distance_scores = np.einsum("ij,ik->ij", spectrogram_matrix, spectrogram_matrix) / (
                        np.linalg.norm(spectrogram_matrix, axis=1)[:, None] * np.linalg.norm(spectrogram_matrix, axis=1)[:, None]
                    )
                    distance_scores_n = distance_scores / (np.max(distance_scores) - np.min(distance_scores))

                    distance_scores_n_reversed = distance_scores_n

                    for threshold in threshold_range:
                        distance_scores_n_reversed_filtered = np.where(distance_scores_n < threshold, 0, distance_scores_n)

                        zero_columns = np.all(distance_scores_n_reversed_filtered == 0, axis=0)
                        zero_column_indices = np.where(zero_columns)[0]

                        if len(zero_column_indices) > 0:
                            start_index = 0
                            while start_index < len(zero_column_indices) and zero_column_indices[start_index] == start_index:
                                start_index += 1

                            end_index = len(zero_column_indices) - 1
                            while end_index >= 0 and zero_column_indices[end_index] == (len(distance_scores_n_reversed_filtered[0]) - 1 - (len(zero_column_indices) - end_index - 1)):
                                end_index -= 1

                            if start_index <= end_index:
                                central_zero_columns_indices = zero_column_indices[start_index:end_index + 1]

                                num_groups = 0
                                if len(central_zero_columns_indices) > 0:
                                    last_index = central_zero_columns_indices[0]
                                    num_groups = 1
                                    for index in central_zero_columns_indices[1:]:
                                        if index > last_index + 1:
                                            num_groups += 1
                                        last_index = index

                                txt_file.write(f"  Processing with hop_length = {hop_length}\n")
                                txt_file.write(f"    Processing with n_mels = {n_mels}\n")
                                txt_file.write(f"      Threshold = {threshold}\n")
                                txt_file.write(f"      Found {num_groups} groups of zero columns in the central range.\n")
                                row = [audio_file, n_mels, hop_length, threshold, num_groups]
                            else:
                                txt_file.write(f"    No zero columns in the central range.\n")
                                row = [audio_file, n_mels, hop_length, threshold, "0"]
                        else:
                            txt_file.write(f"    No zero columns found.\n")
                            row = [audio_file, n_mels, hop_length, threshold, "No zero columns found."]
                        
                        csv_writer.writerow(row)

                        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

                        axs[0, 0].imshow(librosa.power_to_db(spectrogram, ref=np.max), cmap='inferno', aspect='auto', origin='lower')
                        axs[0, 0].set_title(f"Mel-scaled spectrogram={n_mels}")

                        axs[0, 1].imshow(distance_scores_n_reversed, cmap='jet', aspect='auto', origin='lower')
                        axs[0, 1].set_title("Cosine Distance")

                        combined_img = distance_scores_n_reversed_filtered.copy()
                        if start_index <= end_index:
                            for index in central_zero_columns_indices:
                                group_indices = np.where(zero_column_indices == index)[0]
                                if len(group_indices) > 0:
                                    group_start_index = group_indices[0]
                                    group_end_index = group_indices[-1]
                                    group_indices = zero_column_indices[group_start_index:group_end_index + 1]

                                    if len(group_indices) > 0:
                                        mid_index = len(group_indices) // 2
                                        selected_index = group_indices[mid_index]
                                        combined_img[:, selected_index] = 1

                        axs[1, 0].imshow(combined_img, cmap='jet', aspect='auto', origin='lower')
                        axs[1, 0].set_title("Image with Threshold and Segments")

                        axs[1, 1].imshow(librosa.power_to_db(spectrogram, ref=np.max), cmap='inferno', aspect='auto', origin='lower')
                        if start_index <= end_index:
                            for index in central_zero_columns_indices:
                                group_indices = np.where(zero_column_indices == index)[0]
                                if len(group_indices) > 0:
                                    group_start_index = group_indices[0]
                                    group_end_index = group_indices[-1]
                                    group_indices = zero_column_indices[group_start_index:group_end_index + 1]

                                    if len(group_indices) > 0:
                                        mid_index = len(group_indices) // 2
                                        selected_index = group_indices[mid_index]
                                        axs[1, 1].axvline(x=selected_index, color='white', linestyle='--', linewidth=1)

                        axs[1, 1].set_title("Spectrogram with Segments")

                        plt.tight_layout()
                        plt.savefig(f"{audio_file}_n_mels_{n_mels}_hop_{hop_length}_threshold_{threshold}.png")
                        plt.close()
