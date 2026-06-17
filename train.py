import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model(input_shape=(64, 64, 3), num_classes=26):
    """
    Membangun arsitektur model CNN sesuai dengan rancangan Tugas Akhir.
    """
    model = models.Sequential([
        # Input Layer & Conv Block 1
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten & Fully Connected
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5), # Mencegah overfitting
        
        # Output Layer (Softmax)
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Pelatihan Model CNN untuk Klasifikasi Bahasa Isyarat ASL")
    parser.add_argument('--dataset_dir', type=str, default='./dataset', help='Direktori utama dataset ASL Alphabet')
    parser.add_argument('--epochs', type=int, default=15, help='Jumlah epoch pelatihan')
    parser.add_argument('--batch_size', type=int, default=32, help='Ukuran batch size')
    parser.add_argument('--img_size', type=int, default=64, help='Ukuran citra (lebar/tinggi)')
    args = parser.parse_args()
    
    dataset_path = args.dataset_dir
    img_size = args.img_size
    batch_size = args.batch_size
    epochs = args.epochs
    
    print(f"=== Memulai Pelatihan Model CNN ===")
    print(f"Direktori Dataset: {dataset_path}")
    print(f"Ukuran Input: {img_size}x{img_size}x3")
    print(f"Epoch: {epochs} | Batch Size: {batch_size}")
    
    if not os.path.exists(dataset_path):
        print(f"\n[PERINGATAN] Direktori dataset '{dataset_path}' tidak ditemukan.")
        print("Pastikan Anda telah mengunduh 'ASL Alphabet Dataset' dari Kaggle.")
        print("Struktur folder yang diharapkan:")
        print("  dataset/")
        print("    A/")
        print("    B/")
        print("    ... (dst hingga Z)")
        return

    # Augmentasi Data untuk Data Pelatihan & Rescaling
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False, # Bahasa isyarat tangan kanan/kiri berbeda arti
        validation_split=0.2 # Pemisahan data latih & validasi (80% / 20%)
    )
    
    # Generator untuk data latih
    print("\nLoading data latih...")
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Generator untuk data validasi
    print("\nLoading data validasi...")
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    num_classes = train_generator.num_classes
    print(f"\nJumlah Kelas Terdeteksi: {num_classes}")
    
    # Membangun model CNN
    model = build_model(input_shape=(img_size, img_size, 3), num_classes=num_classes)
    model.summary()
    
    # Kompilasi Model
    model.compile(
        optimizer='adam',
        loss='categorical_cross_entropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callback untuk menyimpan model terbaik & early stopping
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('model_cnn_asl.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
    ]
    
    # Memulai training
    print("\nMemulai proses fitting model...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    print("\n=== Pelatihan Selesai! ===")
    print("Model terbaik telah disimpan sebagai 'model_cnn_asl.h5'.")
    
    # Plotting Hasil Akurasi dan Loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Akurasi Model')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('grafik_pelatihan.png')
    print("Grafik hasil pelatihan disimpan sebagai 'grafik_pelatihan.png'.")

if __name__ == '__main__':
    main()
