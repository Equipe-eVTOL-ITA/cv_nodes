import cv2 as cv
import numpy as np
import os

# ── Configuration ────────────────────────────────────────────────────────────
TEMPLATE_SIZE = 400          # Side length (px) for the canonical square
BINARY_THRESHOLD = 100       # Grayscale → binary threshold
DIFF_BLUR_KSIZE = 7          # Gaussian blur kernel for diff cleanup
DIFF_THRESHOLD = 30          # Threshold applied after blur on the diff image

# Calibration angles
ANGLE_AT_0   = 225.0         
ANGLE_AT_100 = -45.0         

def order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   
    rect[2] = pts[np.argmax(s)]   
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   
    rect[3] = pts[np.argmax(d)]   
    return rect

def find_square(binary):
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 1000:
            continue
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4 and area > best_area:
            best = approx
            best_area = area
    return order_corners(best.reshape(4, 2)) if best is not None else None

def warp_to_square(gray, corners, size):
    dst = np.array([[0,0], [size-1,0], [size-1,size-1], [0,size-1]], dtype="float32")
    M = cv.getPerspectiveTransform(corners.astype("float32"), dst)
    return cv.warpPerspective(gray, M, (size, size))

def get_circle_params(binary):
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best_radius, best_center, max_area = None, None, 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 5000: continue
        peri = cv.arcLength(cnt, True)
        if peri == 0: continue
        circularity = (4 * np.pi * area) / (peri * peri)
        if circularity > 0.75 and area > max_area:
            max_area = area
            best_radius = np.sqrt(area / np.pi)
            M = cv.moments(cnt)
            if M["m00"] != 0:
                best_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return best_radius, best_center

def adjust_template(template, target_radius, current_radius):
    if target_radius is None or current_radius is None: return template
    scale = target_radius / current_radius
    h, w = template.shape
    new_h, new_w = int(h * scale), int(w * scale)
    if new_h <= 0 or new_w <= 0: return template
    resized = cv.resize(template, (new_w, new_h), interpolation=cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR)
    result = np.ones((h, w), dtype=np.uint8) * 255
    if scale < 1:
        dy, dx = (h - new_h) // 2, (w - new_w) // 2
        result[dy:dy+new_h, dx:dx+new_w] = resized
    else:
        dy, dx = (new_h - h) // 2, (new_w - w) // 2
        result[0:min(h, new_h-dy), 0:min(w, new_w-dx)] = resized[dy:dy+h, dx:dx+w]
    return result

def isolate_pointer(warped, template):
    diff = cv.subtract(template, warped)
    blurred = cv.GaussianBlur(diff, (DIFF_BLUR_KSIZE, DIFF_BLUR_KSIZE), 0)
    _, mask = cv.threshold(blurred, DIFF_THRESHOLD, 255, cv.THRESH_BINARY)
    return diff, mask

def pointer_angle_pca(mask):
    coords = cv.findNonZero(mask)
    if coords is None or len(coords) < 10: return None, None
    coords = coords.reshape(-1, 2).astype(np.float64)
    mean, eigenvectors = cv.PCACompute(coords, mean=None)
    cx, cy = mean[0]
    vx, vy = eigenvectors[0]
    img_center = np.array([mask.shape[1] / 2.0, mask.shape[0] / 2.0])
    projections = (coords - mean) @ eigenvectors[0]
    pos_mask = projections > 0
    neg_mask = ~pos_mask
    if pos_mask.any() and neg_mask.any():
        if np.linalg.norm(coords[neg_mask].mean(axis=0) - img_center) > np.linalg.norm(coords[pos_mask].mean(axis=0) - img_center):
            vx, vy = -vx, -vy
    return np.degrees(np.arctan2(-vy, vx)), (cx, cy)

def angle_to_pressure(angle_deg):
    sweep = ANGLE_AT_0 - ANGLE_AT_100
    return np.clip(((ANGLE_AT_0 - angle_deg) / sweep) * 100.0, 0.0, 100.0)

def main():
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    template_path = os.path.join(assets_dir, 'template2.png')
    template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
    if template is None: raise FileNotFoundError(f"Template not found: {template_path}")
    template = cv.resize(template, (TEMPLATE_SIZE, TEMPLATE_SIZE))
    _, template_bin = cv.threshold(template, BINARY_THRESHOLD, 255, cv.THRESH_BINARY)
    template_radius, _ = get_circle_params(template_bin)

    camera = cv.VideoCapture(0)
    if not camera.isOpened(): raise RuntimeError("Cannot open camera")

    print("Press 'd' to quit.")
    while True:
        ret, frame = camera.read()
        if not ret: break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, BINARY_THRESHOLD, 255, cv.THRESH_BINARY)
        corners = find_square(binary)
        if corners is not None:
            for i in range(4):
                cv.line(frame, tuple(corners[i].astype(int)), tuple(corners[(i + 1) % 4].astype(int)), (0, 255, 0), 2)
            warped = warp_to_square(gray, corners, TEMPLATE_SIZE)
            _, warped_bin = cv.threshold(warped, BINARY_THRESHOLD, 255, cv.THRESH_BINARY)
            
            best_diff_val, best_mask, best_warped_bin, best_diff_img = -1, None, None, None
            warped_radius, _ = get_circle_params(warped_bin)
            adj_template_bin = adjust_template(template_bin, warped_radius, template_radius)
            
            for rot in range(4):
                rotated_warped = np.rot90(warped_bin, rot)
                diff, mask = isolate_pointer(rotated_warped, adj_template_bin)
                diff_val = np.sum(diff)
                if diff_val > best_diff_val:
                    best_diff_val, best_mask, best_warped_bin, best_diff_img = diff_val, mask, rotated_warped, diff

            if best_mask is not None:
                r_mask, c_mask = get_circle_params(best_warped_bin)
                if r_mask is not None and c_mask is not None:
                    c_mask_img = np.zeros_like(best_mask)
                    cv.circle(c_mask_img, c_mask, int(r_mask), 255, -1)
                    pointer_mask = cv.bitwise_and(best_mask, c_mask_img)
                else:
                    pointer_mask = best_mask

                angle, centroid = pointer_angle_pca(pointer_mask)
                if angle is not None:
                    pressure = angle_to_pressure(angle)
                    cx, cy = int(centroid[0]), int(centroid[1])
                    length = 80
                    ex, ey = int(cx + length * np.cos(np.radians(angle))), int(cy - length * np.sin(np.radians(angle)))
                    warped_vis = cv.cvtColor(best_warped_bin, cv.COLOR_GRAY2BGR)
                    cv.arrowedLine(warped_vis, (cx, cy), (ex, ey), (0, 0, 255), 2)
                    cv.putText(warped_vis, f"Pressure: {pressure:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv.imshow('warped + pointer', warped_vis)
                    cv.imshow('diff', best_diff_img)
                    cv.imshow('pointer mask', pointer_mask)

        cv.imshow('webcam', frame)
        if cv.waitKey(20) & 0xFF == ord('d'): break

    camera.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
