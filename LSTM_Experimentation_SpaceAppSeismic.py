import numpy as np
import pandas as pd
from obspy import read
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import os
import matplotlib.pyplot as plt
import logging
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_logging(log_dir='logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'seismic_model_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to file: {log_file}")

def load_miniseed(file_path):
    logging.info(f"Loading miniseed file: {file_path}")
    st = read(file_path)
    tr = st[0]
    logging.info(f"Loaded data with {len(tr.data)} samples and sampling rate of {tr.stats.sampling_rate} Hz")
    return tr.data, tr.stats.sampling_rate

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    low = max(0.001, min(low, 0.99))  # Ensure low is within (0, 1)
    high = max(low + 0.001, min(high, 0.99))  # Ensure high is within (low, 1)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def calculate_aic(data):
    """Calculate Akaike Information Criterion"""
    n = len(data)
    k = np.arange(1, n-1)
    var_left = np.array([np.var(data[:i]) if i > 1 else np.var(data[:2]) for i in k])
    var_right = np.array([np.var(data[i:]) if i < n-2 else np.var(data[n-2:]) for i in k])
    aic = (k * np.log(var_left + 1e-8) + (n - k - 1) * np.log(var_right + 1e-8))
    return aic

def detect_p_wave(data, sample_rate):
    try:
        # Try to apply bandpass filter
        filtered_data = butter_bandpass_filter(data, 1, min(20, sample_rate/2 - 1), sample_rate)
    except Exception as e:
        logging.warning(f"Bandpass filter failed: {str(e)}. Using original data.")
        filtered_data = data
    
    # Calculate AIC
    aic = calculate_aic(filtered_data)
    
    # Find the minimum of AIC, which corresponds to the P-wave arrival
    p_arrival = np.argmin(aic) + 1
    p_time = p_arrival / sample_rate
    
    logging.info(f"Detected P-wave arrival at {p_time:.2f} seconds (sample {p_arrival})")
    return p_arrival, p_time

def extract_features(data, sample_rate, p_arrival=None):
    window_size = int(10 * sample_rate)  # 10-second window
    step_size = int(5 * sample_rate)  # 5-second step for sliding window
    
    features = []
    for i in range(0, len(data) - window_size, step_size):
        window = data[i:i+window_size]
        
        # Time domain features
        time_features = [
            np.mean(window),
            np.std(window),
            np.max(np.abs(window)),
            np.sum(window**2),  # Energy
            np.mean(np.abs(np.diff(window))),  # Average absolute amplitude change
            np.percentile(window, 25),  # 1st quartile
            np.percentile(window, 75),  # 3rd quartile
            np.median(window),
            np.max(window) - np.min(window),  # Range
            kurtosis(window),  # Kurtosis
            skew(window),  # Skewness
        ]
        
        # Frequency domain features
        fft_vals = np.abs(np.fft.fft(window))
        freq = np.fft.fftfreq(len(window), 1/sample_rate)
        pos_freq_idx = np.where(freq > 0)[0]
        freqs = freq[pos_freq_idx]
        fft_vals = fft_vals[pos_freq_idx]
        
        freq_features = [
            np.sum(fft_vals),  # Total spectral energy
            freqs[np.argmax(fft_vals)],  # Dominant frequency
            np.mean(fft_vals),  # Average spectral amplitude
            np.std(fft_vals),  # Spectral standard deviation
        ]
        
        # Combine all features
        combined_features = time_features + freq_features
        
        # Add a feature indicating proximity to P-wave arrival if provided
        if p_arrival is not None:
            distance_to_p = (i - p_arrival) / sample_rate
            combined_features.append(distance_to_p)
        else:
            combined_features.append(0)  # Use 0 for non-event windows
        
        features.append(combined_features)
    
    features = np.array(features)
    logging.info(f"Extracted {features.shape[0]} windows of features, each with {features.shape[1]} features")
    return features

def prepare_data(data_directory, catalog):
    logging.info("Preparing data from multiple files")
    X = []
    y = []
    file_names = []
    missing_files = []
    max_sequence_length = 0

    for index, row in catalog.iterrows():
        file_path = os.path.join(data_directory, f"{row['filename']}.mseed")
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            logging.warning(f"File not found: {file_path}")
            continue

        try:
            logging.info(f"Processing file {index + 1}/{len(catalog)}: {file_path}")
            data, sample_rate = load_miniseed(file_path)
            
            file_duration = len(data) / sample_rate
            logging.info(f"File duration: {file_duration:.2f} seconds")
            
            p_arrival, p_time = detect_p_wave(data, sample_rate)
            
            # Extract features for a window around the event
            event_start = max(0, p_arrival - int(60 * sample_rate))  # 1 minute before P-arrival
            event_end = min(len(data), p_arrival + int(5 * 60 * sample_rate))  # 5 minutes after P-arrival
            event_data = data[event_start:event_end]
            event_features = extract_features(event_data, sample_rate, p_arrival - event_start)
            logging.info(f"Extracted event features shape: {event_features.shape}")
            
            if event_features.shape[0] > 0:
                max_sequence_length = max(max_sequence_length, event_features.shape[0])
                X.append(event_features)
                y.append(1)  # Known event
                file_names.append(row['filename'])

                # Create non-event sample
                # Use a random 6-minute window that doesn't overlap with the event window
                non_event_duration = int(6 * 60 * sample_rate)
                valid_start = 0
                valid_end = max(0, event_start - non_event_duration)
                if valid_end == 0:
                    valid_start = event_end
                    valid_end = len(data) - non_event_duration

                if valid_end > valid_start:
                    non_event_start = np.random.randint(valid_start, valid_end)
                    non_event_data = data[non_event_start:non_event_start + non_event_duration]
                    non_event_features = extract_features(non_event_data, sample_rate)
                    logging.info(f"Extracted non-event features shape: {non_event_features.shape}")
                    
                    if non_event_features.shape[0] > 0:
                        max_sequence_length = max(max_sequence_length, non_event_features.shape[0])
                        X.append(non_event_features)
                        y.append(0)  # Non-event
                        file_names.append(f"{row['filename']}_non_event")
                    else:
                        logging.warning(f"Skipping non-event for {row['filename']} due to empty features")
                else:
                    logging.warning(f"Skipping non-event for {row['filename']} due to insufficient non-event data")
            else:
                logging.warning(f"Skipping {row['filename']} due to empty features")
            
            # Visualize the P-wave detection and event/non-event windows
            visualize_p_wave_detection(data, sample_rate, p_arrival, row['filename'])
            
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            missing_files.append(file_path)

        # Print progress
        if (index + 1) % 10 == 0 or (index + 1) == len(catalog):
            logging.info(f"Processed {index + 1}/{len(catalog)} files")

    logging.info(f"Max sequence length: {max_sequence_length}")

    if not X:
        logging.error("No files were successfully processed. Check your data and file paths.")
        return np.array([]), np.array([]), [], missing_files

    # Pad or truncate sequences to have the same length
    X_padded = []
    for features in X:
        if features.shape[0] < max_sequence_length:
            padded = np.pad(features, ((0, max_sequence_length - features.shape[0]), (0, 0)), mode='constant')
        else:
            padded = features[:max_sequence_length]
        X_padded.append(padded)

    X = np.array(X_padded)
    y = np.array(y)
    
    logging.info(f"Data preparation complete. X shape: {X.shape}, y shape: {y.shape}")
    
    if missing_files:
        logging.warning(f"Total missing files: {len(missing_files)}")
        logging.warning(f"Missing files: {missing_files}")
    
    return X, y, file_names, missing_files

def visualize_p_wave_detection(data, sample_rate, p_arrival, file_name):
    # Calculate event and non-event windows
    event_start = max(0, p_arrival - int(60 * sample_rate))
    event_end = min(len(data), p_arrival + int(5 * 60 * sample_rate))
    
    # Select a random non-event window
    non_event_duration = int(6 * 60 * sample_rate)
    valid_start = 0
    valid_end = max(0, event_start - non_event_duration)
    if valid_end == 0:
        valid_start = event_end
        valid_end = len(data) - non_event_duration
    
    if valid_end > valid_start:
        non_event_start = np.random.randint(valid_start, valid_end)
        non_event_end = non_event_start + non_event_duration
    else:
        non_event_start = non_event_end = None

    # Create the plot
    plt.figure(figsize=(20, 10))
    time = np.arange(len(data)) / sample_rate / 3600  # Convert to hours
    plt.plot(time, data, 'b-', linewidth=0.5, alpha=0.7)
    plt.title(f"Seismic Activity - {file_name}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Amplitude")
    
    # Mark P-wave arrival
    p_time = p_arrival / sample_rate / 3600
    plt.axvline(x=p_time, color='r', linestyle='--', label='P-wave arrival')
    
    # Mark event window
    plt.axvspan(event_start/sample_rate/3600, event_end/sample_rate/3600, color='y', alpha=0.3, label='Event window')
    
    # Mark non-event window if available
    if non_event_start is not None and non_event_end is not None:
        plt.axvspan(non_event_start/sample_rate/3600, non_event_end/sample_rate/3600, color='g', alpha=0.3, label='Non-event window')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_name}_p_wave_detection.png")
    plt.close()
    logging.info(f"Saved P-wave detection visualization: {file_name}_p_wave_detection.png")

    # Create a zoomed-in plot of the event window
    plt.figure(figsize=(15, 6))
    event_time = np.arange(event_end - event_start) / sample_rate / 60  # Convert to minutes
    event_data = data[event_start:event_end]
    plt.plot(event_time, event_data, 'b-', linewidth=0.5)
    plt.title(f"Seismic Event - {file_name}")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Amplitude")
    plt.axvline(x=(p_arrival - event_start)/sample_rate/60, color='r', linestyle='--', label='P-wave arrival')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_name}_event_zoomed.png")
    plt.close()
    logging.info(f"Saved zoomed event visualization: {file_name}_event_zoomed.png")

class SeismicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SeismicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.relu(self.fc1(h_n[-1]))
        x = self.fc2(x)
        return torch.sigmoid(x)

class SeismicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model(train_loader, model, criterion, optimizer, num_epochs=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 5 == 0:
                logging.info(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                             f'Loss: {loss.item():.4f}, Accuracy: {100 * train_correct / train_total:.2f}%')
        
        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        scheduler.step(train_loss)
        
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], '
                     f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f'Saved best model with loss: {best_loss:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def evaluate_model(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels)
    
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    
    roc_auc = roc_auc_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    logging.info(f"Model evaluation complete. ROC AUC Score: {roc_auc:.4f}")
    return roc_auc

def detect_events(model, data, sample_rate, scaler, threshold=0.5):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    p_arrival, _ = detect_p_wave(data, sample_rate)
    features = extract_features(data, sample_rate, p_arrival)
    scaled_features = scaler.transform(features)
    
    events = []
    window_size = int(2 * sample_rate)  # 2-second window
    step_size = int(0.5 * sample_rate)  # 0.5-second step
    
    for i in range(0, len(scaled_features) - window_size, step_size):
        window = scaled_features[i:i+window_size]
        features_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(features_tensor).item()
        
        if prediction > threshold:
            events.append((i / sample_rate, prediction))
    
    # Merge close events
    merged_events = []
    for event_time, probability in events:
        if not merged_events or event_time - merged_events[-1][0] > 5:  # 5 seconds between events
            merged_events.append((event_time, probability))
        else:
            # Update the probability if it's higher
            if probability > merged_events[-1][1]:
                merged_events[-1] = (event_time, probability)
    
    logging.info(f"Detected {len(merged_events)} events")
    return merged_events

def visualize_seismic_activity(data, sample_rate, events, file_name):
    time = np.arange(len(data)) / sample_rate
    plt.figure(figsize=(15, 6))
    plt.plot(time, data, 'b-', linewidth=0.5, alpha=0.7)
    plt.title(f"Seismic Activity - {file_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    
    for event_time, probability in events:
        plt.axvline(x=event_time, color='r', linestyle='--', alpha=0.5)
        plt.text(event_time, plt.ylim()[1], f'{probability:.2f}', rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{file_name}_seismic_activity.png")
    plt.close()
    logging.info(f"Saved seismic activity visualization: {file_name}_seismic_activity.png")

def main():
    setup_logging()
    logging.info("Starting advanced seismic detection pipeline with LSTM")
    base_directory = "/Users/utkarsh/nasa/data/lunar/"
    train_data_directory = os.path.join(base_directory, "training/data/S12_GradeA/")
    train_catalog_path = os.path.join(base_directory, "training/catalogs/apollo12_catalog_GradeA_final.csv")
    test_data_directory = os.path.join(base_directory, "test/data/S12_GradeB")
    
    train_catalog = pd.read_csv(train_catalog_path)
    
    logging.info("Preparing training data...")
    X_train, y_train, file_names, missing_files = prepare_data(train_data_directory, train_catalog)

    if len(X_train) == 0:
        logging.error("No data to train on. Exiting.")
        return

    positive_samples = sum(y_train)
    total_samples = len(y_train)
    logging.info(f"Positive samples: {positive_samples}/{total_samples} ({positive_samples/total_samples:.2%})")
    
    if missing_files:
        logging.info(f"Missing files: {missing_files}")
    
    # Apply scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    
    train_dataset = SeismicDataset(X_train_scaled, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    input_size = X_train.shape[2]  # Number of features per time step
    hidden_size = 128
    num_layers = 3
    output_size = 1
    
    model = SeismicLSTM(input_size, hidden_size, num_layers, output_size)
    logging.info(f"Created LSTM model with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}")
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    logging.info("Starting model training...")
    model = train_model(train_loader, model, criterion, optimizer, num_epochs=100)
    logging.info("Model training completed")
    
    # Evaluate the model on the training set
    logging.info("Evaluating model on training set...")
    train_roc_auc = evaluate_model(model, train_loader)
    logging.info(f"Training ROC AUC: {train_roc_auc:.4f}")
    
    logging.info("Detecting events in test files...")
    test_results = []
    for file_name in os.listdir(test_data_directory):
        if file_name.endswith(".mseed"):
            file_path = os.path.join(test_data_directory, file_name)
            try:
                logging.info(f"Processing test file: {file_name}")
                data, sample_rate = load_miniseed(file_path)
                
                events = detect_events(model, data, sample_rate, scaler, threshold=0.5)
                logging.info(f"Detected {len(events)} events in {file_name}")
                
                test_results.append({
                    'file_name': file_name,
                    'events': events
                })
                
                print(f"\nDetected events in {file_name}:")
                if events:
                    for event_time, probability in events:
                        print(f"  Event at {event_time:.2f} seconds (probability: {probability:.4f})")
                    
                    # Visualize the seismic activity
                    visualize_seismic_activity(data, sample_rate, events, file_name[:-6])  # Remove '.mseed' from filename
                else:
                    print("  No events detected")
            except Exception as e:
                logging.error(f"Error processing file {file_name}: {str(e)}")

    # Summarize test results
    logging.info("Test set evaluation summary:")
    total_events = sum(len(result['events']) for result in test_results)
    files_with_events = sum(1 for result in test_results if result['events'])
    logging.info(f"Total events detected: {total_events}")
    logging.info(f"Files with detected events: {files_with_events}/{len(test_results)}")

    # Create a summary plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(test_results)), [len(result['events']) for result in test_results])
    plt.xlabel('Test File Index')
    plt.ylabel('Number of Detected Events')
    plt.title('Events Detected in Test Files')
    plt.savefig('test_set_summary.png')
    plt.close()
    logging.info("Saved test set summary plot: test_set_summary.png")

    logging.info("Pipeline completed.")

if __name__ == "__main__":
    main()
