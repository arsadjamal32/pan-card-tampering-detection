import matplotlib.pyplot as plt
import numpy as np

# Simple training curves plot banao
epochs = range(1, 21)

# Sample data (aapke actual training jaisi)
train_acc = [0.55, 0.61, 0.65, 0.68, 0.71, 0.73, 0.75, 0.77, 0.78, 0.79,
             0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.87, 0.88]
val_acc   = [0.52, 0.58, 0.62, 0.65, 0.68, 0.70, 0.71, 0.72, 0.72, 0.73,
             0.73, 0.74, 0.74, 0.74, 0.75, 0.75, 0.75, 0.75, 0.71, 0.71]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs, train_acc, label='Train Accuracy', color='blue')
axes[0].plot(epochs, val_acc,   label='Val Accuracy',   color='orange')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

train_auc = [0.58, 0.64, 0.68, 0.72, 0.75, 0.77, 0.79, 0.81, 0.82, 0.83,
             0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.89, 0.90, 0.90, 0.91]
val_auc   = [0.55, 0.61, 0.65, 0.69, 0.72, 0.74, 0.75, 0.76, 0.77, 0.77,
             0.78, 0.78, 0.79, 0.79, 0.79, 0.80, 0.80, 0.80, 0.80, 0.80]

axes[1].plot(epochs, train_auc, label='Train AUC', color='blue')
axes[1].plot(epochs, val_auc,   label='Val AUC',   color='orange')
axes[1].set_title('AUC Score')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AUC')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('PAN Card Tampering Detection — Training Results')
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
print("training_curves.png saved!")