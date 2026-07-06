import cv2
import numpy as np
from pathlib import Path
import shutil
import random

def augment_image(img):
    """Ek image pe random augmentation apply karo"""
    augmented = img.copy()
    h, w = augmented.shape[:2]

    # 1. Random rotation (sirf thoda — PAN card zyada tilt nahi hota)
    if random.random() > 0.5:
        angle = random.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        augmented = cv2.warpAffine(augmented, M, (w, h),
                                   borderMode=cv2.BORDER_REFLECT)

    # 2. Brightness change
    if random.random() > 0.4:
        factor = random.uniform(0.7, 1.3)
        augmented = np.clip(augmented.astype(np.float32) * factor,
                           0, 255).astype(np.uint8)

    # 3. Slight zoom
    if random.random() > 0.5:
        scale = random.uniform(0.92, 1.08)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(augmented, (new_w, new_h))
        if scale > 1:
            x = (new_w - w) // 2
            y = (new_h - h) // 2
            augmented = resized[y:y+h, x:x+w]
        else:
            augmented = np.zeros_like(img)
            x = (w - new_w) // 2
            y = (h - new_h) // 2
            augmented[y:y+new_h, x:x+new_w] = resized

    # 4. Slight blur (camera shake simulate)
    if random.random() > 0.6:
        ksize = random.choice([3, 5])
        augmented = cv2.GaussianBlur(augmented, (ksize, ksize), 0)

    # 5. Contrast adjustment
    if random.random() > 0.5:
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-15, 15)
        augmented = np.clip(alpha * augmented.astype(np.float32) + beta,
                           0, 255).astype(np.uint8)

    return augmented


def augment_folder(input_dir, output_dir, target_count):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(exist_ok=True)

    # Original images copy karo pehle
    originals = [f for f in input_path.glob('*')
                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    for img in originals:
        shutil.copy(img, output_path / img.name)

    current = len(originals)
    needed  = target_count - current

    print(f"Original images : {current}")
    print(f"Target count    : {target_count}")
    print(f"Need to generate: {needed}")
    print("-" * 40)

    if needed <= 0:
        print("Already enough images!")
        return current

    per_image = (needed // current) + 1
    generated = 0

    for img_path in originals:
        if generated >= needed:
            break

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for j in range(per_image):
            if generated >= needed:
                break

            aug = augment_image(img)
            out_name = f"aug_{generated:04d}_{img_path.name}"
            cv2.imwrite(str(output_path / out_name), aug)
            generated += 1

    final = len(list(output_path.glob('*.jpg'))) + \
            len(list(output_path.glob('*.png')))
    print(f"Generated       : {generated} new images")
    print(f"Total final     : {final}")
    return final


def augment_both():
    print("=" * 50)
    print("GENUINE images augmenting...")
    print("=" * 50)
    genuine_total = augment_folder(
        input_dir="genuine_cropped",
        output_dir="genuine_augmented",
        target_count=500
    )

    print()
    print("=" * 50)
    print("TAMPERED images augmenting...")
    print("=" * 50)
    tampered_total = augment_folder(
        input_dir="tampered",
        output_dir="tampered_augmented",
        target_count=500
    )

    print()
    print("=" * 50)
    print("FINAL DATASET SUMMARY")
    print("=" * 50)
    print(f"Genuine images  : {genuine_total}")
    print(f"Tampered images : {tampered_total}")
    print(f"Total dataset   : {genuine_total + tampered_total}")
    print()
    print("Next step: Model training!")
    print("  genuine_augmented/  → class 0 (genuine)")
    print("  tampered_augmented/ → class 1 (tampered)")


augment_both()