import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance
import io, os

print("TensorFlow:", tf.__version__)

# ── ELA function ──
def get_ela(image_path, quality=90, scale=15):
    try:
        orig = Image.open(image_path).convert('RGB')
        buffer = io.BytesIO()
        orig.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)
        ela = ImageChops.difference(orig, compressed)
        extrema = ela.getextrema()
        max_diff = max([e[1] for e in extrema]) or 1
        ela = ImageEnhance.Brightness(ela).enhance(255.0 / max_diff * scale)
        return ela
    except:
        return None

# ── Load images with ELA ──
def load_images_with_ela(folder, label, target_size=(224, 224)):
    images, labels = [], []
    folder = Path(folder)
    files = (list(folder.glob('*.jpg')) +
             list(folder.glob('*.jpeg')) +
             list(folder.glob('*.png')))
    print(f"  {folder.name}: {len(files)} images")

    for img_path in files:
        try:
            # Original RGB
            orig = Image.open(img_path).convert('RGB').resize(target_size)
            orig_arr = np.array(orig) / 255.0

            # ELA
            ela = get_ela(img_path)
            if ela is None:
                continue
            ela_arr = np.array(ela.resize(target_size)) / 255.0

            # Stack: 6 channels (RGB + ELA)
            combined = np.concatenate([orig_arr, ela_arr], axis=-1)
            images.append(combined)
            labels.append(label)
        except Exception as e:
            continue

    return images, labels

print("\nLoading with ELA features...")
g_imgs, g_labels = load_images_with_ela('genuine_augmented', 0)
t_imgs, t_labels = load_images_with_ela('tampered_augmented', 1)

X = np.array(g_imgs + t_imgs)
y = np.array(g_labels + t_labels)

print(f"Total: {len(X)} | Shape: {X.shape}")
print(f"Genuine: {sum(y==0)} | Tampered: {sum(y==1)}")

# ── Split ──
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.11, random_state=42, stratify=y_temp)

print(f"Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}")

# ── Custom CNN for 6-channel input ──
print("\nBuilding CNN with ELA...")

inputs = keras.Input(shape=(224, 224, 6))

# Block 1
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Dropout(0.25)(x)

# Block 2
x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Dropout(0.25)(x)

# Block 3
x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Dropout(0.25)(x)

# Block 4
x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.GlobalAveragePooling2D()(x)

# Classifier
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy',
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

model.summary()

# ── Balanced class weights ──
class_weight = {0: 1.0, 1: 1.5}  # Thoda zyada, bahut nahi

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=12,
        restore_best_weights=True, mode='max'),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, min_lr=1e-7, verbose=1),
    keras.callbacks.ModelCheckpoint(
        'best_pan_model.keras',
        monitor='val_auc',
        save_best_only=True, mode='max', verbose=1)
]

# ── Training ──
print("\nTraining with ELA features...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# ── Evaluate with best threshold ──
print("\nFinding best threshold...")
y_pred_prob = model.predict(X_val).flatten()

best_thresh = 0.5
best_f1 = 0
for thresh in np.arange(0.3, 0.7, 0.05):
    y_p = (y_pred_prob > thresh).astype(int)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_val, y_p, average='macro')
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"Best threshold: {best_thresh:.2f}")

# ── Test Results ──
y_pred_prob_test = model.predict(X_test).flatten()
y_pred = (y_pred_prob_test > best_thresh).astype(int)

print("\n" + "="*50)
print(f"FINAL RESULTS (threshold={best_thresh:.2f})")
print("="*50)
print(classification_report(
    y_test, y_pred,
    target_names=['Genuine', 'Tampered'],
    zero_division=0))

# ── Confusion Matrix ──
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Genuine','Tampered'],
            yticklabels=['Genuine','Tampered'])
plt.title('Confusion Matrix — ELA + CNN')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ── Training Curves ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_title('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['auc'], label='Train AUC')
axes[1].plot(history.history['val_auc'], label='Val AUC')
axes[1].set_title('AUC Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()

# ── Save ──
model.save('pan_tampering_model.keras')
print("\nModel saved!")
print("DONE!")