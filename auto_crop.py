import cv2
import numpy as np
from pathlib import Path
import shutil

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect.astype("float32"), dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def is_valid_pan_card(contour, img_shape):
    """
    Check karo ki ye contour PAN card hai ya nahi
    """
    img_h, img_w = img_shape[:2]
    img_area = img_h * img_w
    
    # Bounding rectangle lo
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h
    
    # 1. Area check — image ka kam se kam 20% hona chahiye
    area_ratio = area / img_area
    if area_ratio < 0.20:
        return False, "too_small"
    
    # 2. Aspect ratio check — PAN card 1.58:1 hota hai (width:height)
    if h == 0:
        return False, "zero_height"
    aspect = w / h
    # Landscape: 1.2 to 2.5 ke beech hona chahiye
    # Portrait: 0.4 to 0.8 (rotated card)
    if not (1.2 <= aspect <= 2.5 or 0.4 <= aspect <= 0.8):
        return False, f"bad_aspect_{aspect:.2f}"
    
    return True, "valid"

def crop_pan_card(image_path, output_path, target_size=(224, 224)):
    img = cv2.imread(str(image_path))
    if img is None:
        return False, "read_error"
    
    original = img.copy()
    h, w = img.shape[:2]
    
    # ── Method 1: Large contour detection ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Different preprocessing try karo
    methods_found = []
    
    # Method 1A: Simple threshold
    _, thresh = cv2.threshold(gray, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        valid, reason = is_valid_pan_card(cnt, img.shape)
        if valid:
            area = cv2.contourArea(cnt)
            methods_found.append((area, cnt, "threshold"))
    
    # Method 1B: Edge detection with large kernel
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 20, 80)
    kernel = np.ones((7, 7), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=3)
    edges_dilated = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
    contours2, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours2:
        valid, reason = is_valid_pan_card(cnt, img.shape)
        if valid:
            area = cv2.contourArea(cnt)
            methods_found.append((area, cnt, "edge"))
    
    # Method 1C: Color based — PAN card light colored hota hai
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Light colors (white/light blue background of PAN)
    lower_light = np.array([0, 0, 150])
    upper_light = np.array([180, 80, 255])
    mask = cv2.inRange(hsv, lower_light, upper_light)
    kernel2 = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)
    contours3, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours3:
        valid, reason = is_valid_pan_card(cnt, img.shape)
        if valid:
            area = cv2.contourArea(cnt)
            methods_found.append((area, cnt, "color"))
    
    # Sabse bada valid contour lo
    best_contour = None
    best_method = None
    if methods_found:
        methods_found.sort(key=lambda x: x[0], reverse=True)
        _, best_contour, best_method = methods_found[0]
    
    if best_contour is not None:
        # 4-point transform try karo
        peri = cv2.arcLength(best_contour, True)
        approx = cv2.approxPolyDP(best_contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            cropped = four_point_transform(original, approx.reshape(4, 2))
        else:
            # Bounding rect use karo
            x, y, bw, bh = cv2.boundingRect(best_contour)
            # Thoda margin add karo
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            bw = min(w - x, bw + 2*margin)
            bh = min(h - y, bh + 2*margin)
            cropped = original[y:y+bh, x:x+bw]
        
        method_used = best_method
    else:
        # ── Fallback: Image ka central 85% lo ──
        margin_h = int(h * 0.075)
        margin_w = int(w * 0.075)
        cropped = original[margin_h:h-margin_h, margin_w:w-margin_w]
        method_used = "fallback"
    
    # Landscape orientation ensure karo
    ch, cw = cropped.shape[:2]
    if ch > cw:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
    
    # Resize
    result = cv2.resize(cropped, target_size)
    cv2.imwrite(str(output_path), result)
    return True, method_used


def process_all(input_dir="genuine", output_dir="genuine_cropped"):
    # Purana output clear karo
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(exist_ok=True)
    
    failed_path = Path("failed_crops")
    if failed_path.exists():
        shutil.rmtree(failed_path)
    failed_path.mkdir(exist_ok=True)
    
    images = [f for f in Path(input_dir).glob('*')
              if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    print(f"Total images: {len(images)}")
    print("-" * 50)
    
    counts = {"threshold": 0, "edge": 0, 
              "color": 0, "fallback": 0, "failed": 0}
    fallback_list = []
    
    for i, img_path in enumerate(images, 1):
        out_path = output_path / img_path.name
        ok, method = crop_pan_card(img_path, out_path)
        
        if ok:
            counts[method] = counts.get(method, 0) + 1
            if method == "fallback":
                fallback_list.append(img_path.name)
                shutil.copy(out_path, failed_path / img_path.name)
            status = f"OK ({method})"
        else:
            counts["failed"] += 1
            status = "FAILED"
        
        print(f"[{i:03d}/{len(images)}] {status:<20} {img_path.name}")
    
    print("\n" + "=" * 50)
    print(f"Threshold method : {counts['threshold']}")
    print(f"Edge method      : {counts['edge']}")
    print(f"Color method     : {counts['color']}")
    print(f"Fallback         : {counts['fallback']}")
    print(f"Failed           : {counts['failed']}")
    print(f"\nTotal processed  : {len(images)}")
    print(f"\nCropped images   : '{output_dir}/'")
    if fallback_list:
        print(f"Manually review  : 'failed_crops/' ({len(fallback_list)} images)")

process_all()