import cv2 as cv
import numpy as np
import os

# ── Configuration ────────────────────────────────────────────────────────────
TEMPLATE_SIZE = 400          # Side length (px) for the canonical square
BINARY_THRESHOLD = 100       # Grayscale → binary threshold
DIFF_BLUR_KSIZE = 7          # Gaussian blur kernel for diff cleanup
DIFF_THRESHOLD = 30          # Threshold applied after blur on the diff image

# Angle (degrees, math convention: 0=right, CCW positive) observed when the
# pointer is at the 0-pressure and 100-pressure marks.
# These MUST be calibrated once with known reference images.
ANGLE_AT_0   = 225.0         # pointer pointing to ~7 o'clock  (bottom-left)
ANGLE_AT_100 = -45.0         # pointer pointing to ~5 o'clock  (bottom-right)
# The sweep goes CW from 225° down to -45° (equivalently 315°), which is 270°.


def order_corners(pts):
    """
    Order 4 corner points as: top-left, top-right, bottom-right, bottom-left.
    Works with any rotation / perspective of the quadrilateral.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left  has smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest x+y
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # top-right has smallest x-y
    rect[3] = pts[np.argmax(d)]   # bottom-left has largest x-y
    return rect


def find_square(binary):
    """
    Find the largest approximate quadrilateral in a binary image.
    Returns the 4 ordered corners or None.
    """
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

    if best is None:
        return None

    return order_corners(best.reshape(4, 2))


def warp_to_square(gray, corners, size):
    """
    Perspective-warp the region defined by `corners` into a `size x size` square.
    """
    dst = np.array([
        [0,      0],
        [size-1, 0],
        [size-1, size-1],
        [0,      size-1],
    ], dtype="float32")

    M = cv.getPerspectiveTransform(corners.astype("float32"), dst)
    warped = cv.warpPerspective(gray, M, (size, size))
    return warped


def isolate_pointer(warped, template):
    """
    Subtract the template from the warped image.
    Returns a binary mask where the pointer pixels are white.
    """
    diff = cv.absdiff(warped, template)
    blurred = cv.GaussianBlur(diff, (DIFF_BLUR_KSIZE, DIFF_BLUR_KSIZE), 0)
    _, mask = cv.threshold(blurred, DIFF_THRESHOLD, 255, cv.THRESH_BINARY)
    return diff, mask


def pointer_angle_pca(mask):
    """
    Use PCA on the white pixels of `mask` to find the dominant direction.
    Disambiguates the 180° PCA ambiguity by picking the direction that
    points AWAY from the image center (pointer tip extends outward).
    Returns the angle in degrees (math convention: 0=right, CCW positive)
    and the centroid (cx, cy).
    """
    coords = cv.findNonZero(mask)
    if coords is None or len(coords) < 10:
        return None, None

    coords = coords.reshape(-1, 2).astype(np.float64)
    mean, eigenvectors = cv.PCACompute(coords, mean=None)
    cx, cy = mean[0]

    # First eigenvector = direction of maximum variance = pointer axis
    vx, vy = eigenvectors[0]

    # ── Disambiguate 180° ambiguity ──────────────────────────────────────
    # The image center is the hub of the manometer.
    # The pointer TIP is on the side of the centroid that is farther from
    # the image center.  Project each pixel onto the eigenvector; pixels
    # with positive projection are on the +v side, negative on the -v side.
    # The side whose mean is farther from image center is the tip side.
    img_center = np.array([mask.shape[1] / 2.0, mask.shape[0] / 2.0])
    projections = (coords - mean) @ eigenvectors[0]       # scalar per pixel
    pos_mask = projections > 0
    neg_mask = ~pos_mask

    if pos_mask.any() and neg_mask.any():
        mean_pos = coords[pos_mask].mean(axis=0)
        mean_neg = coords[neg_mask].mean(axis=0)
        dist_pos = np.linalg.norm(mean_pos - img_center)
        dist_neg = np.linalg.norm(mean_neg - img_center)
        if dist_neg > dist_pos:
            vx, vy = -vx, -vy  # flip to point toward the tip

    angle_rad = np.arctan2(-vy, vx)  # negate vy because image y-axis is flipped
    angle_deg = np.degrees(angle_rad)

    return angle_deg, (cx, cy)


def angle_to_pressure(angle_deg):
    """
    Map detected pointer angle to pressure value (0–100).
    Uses a linear interpolation between the two calibrated angles.
    """
    # Normalize angles so the sweep is monotonically decreasing
    # from ANGLE_AT_0 (225°) to ANGLE_AT_100 (-45° = 315° CW)
    sweep = ANGLE_AT_0 - ANGLE_AT_100   # total angular sweep (positive)
    fraction = (ANGLE_AT_0 - angle_deg) / sweep
    pressure = fraction * 100.0
    return np.clip(pressure, 0.0, 100.0)


def main():
    # ── Load template ────────────────────────────────────────────────────
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    template_path = os.path.join(assets_dir, 'template.png')
    template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Template not found: {template_path}")

    # Resize template to canonical size and binarize
    template = cv.resize(template, (TEMPLATE_SIZE, TEMPLATE_SIZE))
    _, template_bin = cv.threshold(template, BINARY_THRESHOLD, 255, cv.THRESH_BINARY)

    # ── Open camera ──────────────────────────────────────────────────────
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Cannot open camera")

    print("Press 'd' to quit.")

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, BINARY_THRESHOLD, 255, cv.THRESH_BINARY)

        # ── Step 2: find & warp the white square ─────────────────────────
        corners = find_square(binary)
        if corners is not None:
            # Draw detected square on original frame (for debug)
            for i in range(4):
                pt1 = tuple(corners[i].astype(int))
                pt2 = tuple(corners[(i + 1) % 4].astype(int))
                cv.line(frame, pt1, pt2, (0, 255, 0), 2)

            warped = warp_to_square(gray, corners, TEMPLATE_SIZE)
            _, warped_bin = cv.threshold(warped, BINARY_THRESHOLD, 255, cv.THRESH_BINARY)

            # ── Step 3: template subtraction ─────────────────────────────
            diff, pointer_mask = isolate_pointer(warped_bin, template_bin)

            # ── Step 4: PCA angle ────────────────────────────────────────
            angle, centroid = pointer_angle_pca(pointer_mask)
            if angle is not None:
                pressure = angle_to_pressure(angle)
                cx, cy = int(centroid[0]), int(centroid[1])

                # Draw pointer direction on warped image
                length = 80
                ex = int(cx + length * np.cos(np.radians(angle)))
                ey = int(cy - length * np.sin(np.radians(angle)))
                warped_vis = cv.cvtColor(warped_bin, cv.COLOR_GRAY2BGR)
                cv.arrowedLine(warped_vis, (cx, cy), (ex, ey), (0, 0, 255), 2)
                cv.putText(warped_vis, f"Angle: {angle:.1f} deg", (10, 25),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv.putText(warped_vis, f"Pressure: {pressure:.1f}", (10, 55),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv.imshow('warped + pointer', warped_vis)
                cv.imshow('diff', diff)
                cv.imshow('pointer mask', pointer_mask)

        cv.imshow('webcam', frame)
        cv.imshow('template', template_bin)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break

    camera.release()


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
