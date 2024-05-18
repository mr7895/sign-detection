# Import necessary libraries
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

# Define data path and actions
DATA_PATH = 'MP_Data'
actions = np.array([chr(i) for i in range(ord('A'), ord('Z')+1)])  # Actions A to Z
no_sequences = 30
sequence_length = 30

# Create a mapping from labels to integers
label_map = {label: num for num, label in enumerate(actions)}

# Load sequences and corresponding labels
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            if os.path.exists(file_path):
                res = np.load(file_path, allow_pickle=True)
                if res.shape != (63,):  # Assume each data point must have 63 elements
                    print(f"Error: Incorrect shape {res.shape} in file {file_path}")
                    window = []  # Reset the window if shape is incorrect
                    break
                window.append(res)
            else:
                print(f"Missing file: {file_path}")
                window = []  # Clear the current sequence if any file is missing
                break
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
        else:
            print(f"Incomplete sequence for {action} {sequence}: Expected {sequence_length}, got {len(window)}")

# Check if we have loaded any data
if not sequences:
    raise ValueError("No data available for training.")

# Convert to numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Set up TensorBoard logging
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, X.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[tb_callback])

# Print model summary
model.summary()

# Save the model to disk
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')

print("Model training complete and model saved.")
