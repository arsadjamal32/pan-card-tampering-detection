import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random
import shutil

# ── Random data generators ──
def random_name():
    first = ["RAHUL", "PRIYA", "AMIT", "SUNITA", "RAJESH",
             "NEHA", "VIKRAM", "POOJA", "SURESH", "KAVITA",
             "DEEPAK", "ANITA", "MANOJ", "SUMAN", "ANIL"]
    last  = ["SHARMA", "SINGH", "KUMAR", "GUPTA", "VERMA",
             "MISHRA", "YADAV", "PANDEY", "TIWARI", "JOSHI"]
    return f"{random.choice(first)} {random.choice(last)}"

def random_dob():
    d = random.randint(1, 28)
    m = random.randint(1, 12)
    y = random.randint(1965, 2002)
    return f"{d:02d}/{m:02d}/{y}"

def random_pan():
    import string
    L = string.ascii_uppercase
    D = string.digits
    return (''.join(random.choices(L, k=5)) +
            ''.join(random.choices(D, k=4)) +
            random.choice(L))

# ── 5 Tampering Methods ──

def tamper_type1_text_overlay(img_pil):
    """Name region pe naya naam likho"""
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Name region (top-left area, approximate)
    x1 = int(w * 0.05)
    y1 = int(h * 0.30)
    x2 = int(w * 0.65)
    y2 = int(h * 0.45)

    # Background rectangle — original color se match karne ki koshish
    bg_color = img.getpixel((x1 + 5, y1 + 5))
    if isinstance(bg_color, int):
        bg_color = (bg_color, bg_color, bg_color)
    bg_color = (
        min(255, bg_color[0] + 10),
        min(255, bg_color[1] + 10),
        min(255, bg_color[2] + 10)
    )
    draw.rectangle([x1, y1, x2, y2], fill=bg_color)

    # Naya naam likho
    font_size = max(8, int(h * 0.07))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    draw.text((x1 + 3, y1 + 3), random_name(),
              fill=(10, 10, 120), font=font)
    return img


def tamper_type2_dob_change(img_pil):
    """Date of birth region badlo"""
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    x1 = int(w * 0.05)
    y1 = int(h * 0.50)
    x2 = int(w * 0.50)
    y2 = int(h * 0.65)

    bg_color = img.getpixel((x1 + 5, y1 + 5))
    if isinstance(bg_color, int):
        bg_color = (bg_color, bg_color, bg_color)
    draw.rectangle([x1, y1, x2, y2], fill=bg_color)

    font_size = max(8, int(h * 0.07))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    draw.text((x1 + 3, y1 + 3), random_dob(),
              fill=(10, 10, 120), font=font)
    return img


def tamper_type3_pan_number(img_pil):
    """PAN number badlo"""
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    x1 = int(w * 0.05)
    y1 = int(h * 0.65)
    x2 = int(w * 0.60)
    y2 = int(h * 0.80)

    bg_color = img.getpixel((x1 + 5, y1 + 5))
    if isinstance(bg_color, int):
        bg_color = (bg_color, bg_color, bg_color)
    draw.rectangle([x1, y1, x2, y2], fill=bg_color)

    font_size = max(8, int(h * 0.07))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    draw.text((x1 + 3, y1 + 3), random_pan(),
              fill=(10, 10, 120), font=font)
    return img


def tamper_type4_photo_region(img_pil):
    """Photo area mein noise/color block daal do"""
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Photo region (right side of PAN card)
    x1 = int(w * 0.68)
    y1 = int(h * 0.15)
    x2 = int(w * 0.95)
    y2 = int(h * 0.75)

    # Random color block
    r = random.randint(150, 220)
    g = random.randint(150, 220)
    b = random.randint(150, 220)
    draw.rectangle([x1, y1, x2, y2], fill=(r, g, b))

    # Thodi lines add karo realistic banane ke liye
    for _ in range(3):
        lx = random.randint(x1, x2)
        draw.line([(lx, y1), (lx, y2)],
                  fill=(r-30, g-30, b-30), width=1)
    return img


def tamper_type5_noise_patch(img_pil):
    """Random jagah pe pixel noise patches add karo"""
    img_array = np.array(img_pil)
    h, w = img_array.shape[:2]

    num_patches = random.randint(2, 5)
    for _ in range(num_patches):
        # Random position aur size
        px = random.randint(0, w - 30)
        py = random.randint(0, h - 20)
        pw = random.randint(15, 45)
        ph = random.randint(8, 20)

        # Ensure bounds
        px2 = min(px + pw, w)
        py2 = min(py + ph, h)

        # Gaussian noise
        noise = np.random.normal(
            loc=128, scale=40,
            size=(py2-py, px2-px, 3)
        ).astype(np.uint8)
        img_array[py:py2, px:px2] = noise

    return Image.fromarray(img_array)


# ── Main Generator ──
def generate_tampered_dataset(
        genuine_dir="genuine_cropped",
        output_dir="tampered",
        target_per_type=40):   # har type se 40 images = 200 total

    genuine_path = Path(genuine_dir)
    output_path  = Path(output_dir)

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(exist_ok=True)

    # Subfolders — type wise
    type_dirs = {}
    tamper_types = {
        "type1_name":    tamper_type1_text_overlay,
        "type2_dob":     tamper_type2_dob_change,
        "type3_pan_num": tamper_type3_pan_number,
        "type4_photo":   tamper_type4_photo_region,
        "type5_noise":   tamper_type5_noise_patch,
    }
    for t in tamper_types:
        d = output_path / t
        d.mkdir(exist_ok=True)
        type_dirs[t] = d

    genuine_images = [f for f in genuine_path.glob('*')
                      if f.suffix.lower() in ['.jpg','.jpeg','.png']]

    if not genuine_images:
        print(f"ERROR: '{genuine_dir}/' mein koi image nahi mili!")
        return

    print(f"Genuine images found : {len(genuine_images)}")
    print(f"Target per type      : {target_per_type}")
    print(f"Total tampered target: {target_per_type * len(tamper_types)}")
    print("-" * 50)

    total = 0
    for type_name, tamper_fn in tamper_types.items():
        count = 0
        attempts = 0

        while count < target_per_type and attempts < target_per_type * 3:
            attempts += 1
            src = random.choice(genuine_images)

            try:
                img_pil = Image.open(src).convert('RGB')
                tampered = tamper_fn(img_pil)

                out_name = f"{type_name}_{count:03d}.jpg"
                out_path = type_dirs[type_name] / out_name

                # Main folder mein bhi save karo (flat structure)
                flat_path = output_path / out_name

                tampered.save(str(out_path), quality=85)
                tampered.save(str(flat_path), quality=85)

                count += 1

            except Exception as e:
                print(f"  Skip ({src.name}): {e}")
                continue

        print(f"{type_name:<20} : {count} images")
        total += count

    print("\n" + "=" * 50)
    print(f"Total tampered images: {total}")
    print(f"Saved in             : '{output_dir}/'")
    print(f"Sub-folders          : type1 to type5 (40 each)")


# Run karo
generate_tampered_dataset(
    genuine_dir="genuine_cropped",
    output_dir="tampered",
    target_per_type=40
)