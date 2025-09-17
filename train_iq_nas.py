import os
import sys
import glob
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import keras_tuner as kt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==========================
# 0️⃣ Global classes
# ==========================
CLASSES = ["LTE", "DVB-T", "WiFi"]
PREFIXES = ["lte", "dvbt", "wf"]

# ==========================
# 1️⃣ Utility functions
# ==========================
def read_iq_file(filename):
    data = np.fromfile(filename, dtype=np.float32)
    return data[0::2] + 1j * data[1::2]

def normalize_iq(iq):
    return (iq - np.mean(iq)) / np.std(iq)

def chunks_from_iq(iq, chunk_samples):
    chunks_list = [iq[i:i+chunk_samples] for i in range(0, len(iq), chunk_samples)]
    return [np.column_stack((np.real(c), np.imag(c))) for c in chunks_list if len(c) == chunk_samples]

def load_dataset(base_folder, chunk_samples):
    X_all, y_all = [], []

    for class_name, prefix in zip(CLASSES, PREFIXES):
        pattern = os.path.join(base_folder, f"{prefix}*.bin")
        files = glob.glob(pattern)

        if not files:
            print(f"⚠️ No files found for class {class_name} in {base_folder}")
            continue

        class_index = CLASSES.index(class_name)

        for file in files:
            iq = normalize_iq(read_iq_file(file))
            X_chunks = chunks_from_iq(iq, chunk_samples)
            X_all.extend(X_chunks)
            y_all.extend([class_index] * len(X_chunks))

    if not X_all:
        return None, None

    X_all = np.array(X_all)
    y_all = to_categorical(y_all, num_classes=len(CLASSES))
    return X_all, y_all

# ==========================
# 2️⃣ Model builder for KerasTuner
# ==========================
def model_builder(hp, input_len):
    model = Sequential()

    # Conv layers
    conv_layers = hp.Int('conv_layers', 1, 3)
    filters = hp.Int('conv_filters', 32, 128, step=32)
    for i in range(conv_layers):
        if i == 0:
            model.add(Conv1D(filters, 3, activation='relu', input_shape=(input_len, 2)))
        else:
            model.add(Conv1D(filters * (i+1), 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))

    # LSTM layer
    lstm_units = hp.Int('lstm_units', 32, 128, step=32)
    model.add(LSTM(lstm_units))
    model.add(Dense(lstm_units // 2, activation='relu'))
    model.add(Dropout(hp.Float('dropout_dense', 0.2, 0.5, step=0.1)))
    model.add(Dense(len(CLASSES), activation='softmax'))

    # Compile
    model.compile(
        optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==========================
# 3️⃣ Main
# ==========================
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python train_iq_nas.py <model.keras> <dt_folder> <chunk_samples> <batch_size> <epoch>")
        sys.exit(1)

    model_path = sys.argv[1]
    folder = sys.argv[2]
    chunk_samples = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    epoch = int(sys.argv[5])

    # Load datasets
    X_train, y_train = load_dataset(os.path.join(folder, "train"), chunk_samples)
    X_val, y_val = load_dataset(os.path.join(folder, "validation"), chunk_samples)
    X_test, y_test = load_dataset(os.path.join(folder, "test"), chunk_samples)

    if X_train is None or X_val is None:
        print("⚠️ Not enough data to train")
        sys.exit(1)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # ==========================
    # KerasTuner Search
    # ==========================
    tuner = kt.RandomSearch(
        lambda hp: model_builder(hp, input_len=chunk_samples),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        overwrite=False,  # ✅ no sobrescribe trials completados
        directory='nas_results',
        project_name='cnn_lstm_nas'
    )

    tuner.search_space_summary()

    # ==========================
    # Callbacks
    # ==========================
    os.makedirs(os.path.join('nas_results', 'checkpoints'), exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=os.path.join('nas_results', 'checkpoints', 'epoch_{epoch:02d}_valacc_{val_accuracy:.4f}.h5'),
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

    # Run tuner search
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epoch,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks
    )

    # Get best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Evaluate on test
    if X_test is not None:
        loss, acc = best_model.evaluate(X_test, y_test, verbose=0)
        print(f"📊 Final Test acc={acc:.4f}, loss={loss:.4f}")

    # Save best model
    best_model.save(model_path)
    print(f"💾 Best model saved to {model_path}")
