import shutil
from pathlib import Path

def extract_images_only(roboflow_zip_folder, output_dir="genuine"):
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    
    count = 0
    for split in ['train', 'valid', 'test']:
        img_folder = Path(roboflow_zip_folder) / split / 'images'
        
        if not img_folder.exists():
            print(f"{split}/images/ nahi mili — skip")
            continue
        
        imgs = [i for i in img_folder.glob('*')
                if i.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        for img in imgs:
            shutil.copy(img, output / f"{split}_{img.name}")
            count += 1
        
        print(f"{split}: {len(imgs)} images copy hui")
    
    print(f"\nTotal genuine images ready: {count}")
    print(f"Saved in: '{output_dir}' folder")

extract_images_only(".")