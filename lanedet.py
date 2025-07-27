import cv2
import numpy as np
from collections import deque


# Increased history buffer size for smoother lane detection
left_history = deque(maxlen=15)
right_history = deque(maxlen=15)


def region_of_interest(img):
    """
    Create a mask with a trapezoid shape to focus on the road area.
    The trapezoid is more adaptive to different road conditions.
    """
    height, width = img.shape[:2]
    
    # Define a more adaptive region of interest
    # Bottom width is wider to capture more of the road
    bottom_width_percent = 0.95  # 95% of image width at the bottom
    top_width_percent = 0.5      # 50% of image width at the top
    height_percent = 0.6         # Start at 60% of the image height
    
    bottom_left = (int(width * (1 - bottom_width_percent) / 2), height)
    bottom_right = (int(width * (1 + bottom_width_percent) / 2), height)
    top_left = (int(width * (1 - top_width_percent) / 2), int(height * height_percent))
    top_right = (int(width * (1 + top_width_percent) / 2), int(height * height_percent))
    
    polygons = np.array([[bottom_left, bottom_right, top_right, top_left]])
    
    # Create and apply the mask
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    
    # Optional: Draw the ROI on a debug image if needed
    # debug_img = img.copy()
    # cv2.polylines(debug_img, [polygons], True, 255, 2)
    # cv2.imshow("ROI", debug_img)
    
    return cv2.bitwise_and(img, mask)


def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    if slope == 0: slope = 0.1
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit, right_fit = [], []
    img_center = image.shape[1] // 2
    height, width = image.shape[:2]

    if lines is None:
        # Create default lines if none are detected
        # This ensures we always have some lane lines
        left_slope = -0.8
        right_slope = 0.8
        left_intercept = height - 100
        right_intercept = height - 100
        
        try:
            left_line = make_coordinates(image, (left_slope, left_intercept))
            right_line = make_coordinates(image, (right_slope, right_intercept))
            return left_line, right_line
        except:
            return None, None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 - x1 == 0: continue
        
        # Calculate slope and intercept
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        # Filter out lines with very small slopes (nearly horizontal)
        if abs(slope) < 0.15: continue
        
        # Even more relaxed slope thresholds to capture more lines
        if slope < -0.15 and x1 < img_center + 50:  # Allow lines slightly past center
            # Only include lines with reasonable slopes (not too steep)
            if slope > -2.5:  # More permissive upper bound
                left_fit.append((slope, intercept))
        elif slope > 0.15 and x1 > img_center - 50:  # Allow lines slightly before center
            # Only include lines with reasonable slopes (not too steep)
            if slope < 2.5:  # More permissive upper bound
                right_fit.append((slope, intercept))

    # More robust handling of empty fits with fallback values
    try:
        left_line = make_coordinates(image, np.average(left_fit, axis=0)) if left_fit else None
    except:
        # Fallback for left line if averaging fails
        try:
            if left_fit:
                # Try using the first valid left line instead of averaging
                left_line = make_coordinates(image, left_fit[0])
            else:
                left_line = None
        except:
            left_line = None
        
    try:
        right_line = make_coordinates(image, np.average(right_fit, axis=0)) if right_fit else None
    except:
        # Fallback for right line if averaging fails
        try:
            if right_fit:
                # Try using the first valid right line instead of averaging
                right_line = make_coordinates(image, right_fit[0])
            else:
                right_line = None
        except:
            right_line = None

    return left_line, right_line


def draw_lane_area(image, left_line, right_line):
    lane_image = np.zeros_like(image)

    # Draw lane lines if detected
    if left_line is not None:
        cv2.line(lane_image, tuple(left_line[:2]), tuple(left_line[2:]), (255, 0, 255), 6)
    if right_line is not None:
        cv2.line(lane_image, tuple(right_line[:2]), tuple(right_line[2:]), (255, 0, 255), 6)

    # Fill lane area with green if both lines are detected
    if left_line is not None and right_line is not None:
        # Relaxed condition: only check if left line is generally to the left of right line
        # This makes the green area appear more consistently
        if left_line[0] < right_line[0]:
            try:
                points = np.array([[
                    tuple(left_line[:2]),
                    tuple(left_line[2:]),
                    tuple(right_line[2:]),
                    tuple(right_line[:2])
                ]], dtype=np.int32)
                cv2.fillPoly(lane_image, points, (0, 255, 0))
            except:
                # Fallback if there's an issue with the points
                pass

    return lane_image


def estimate_turn(left, right):
    # More robust turn estimation
    if left is None or right is None:
        return "Detecting..."
    
    try:
        # Calculate the midpoints of the lane at the bottom and top
        mid_bottom = (left[0] + right[0]) // 2
        mid_top = (left[2] + right[2]) // 2
        
        # Calculate the horizontal shift
        delta = mid_top - mid_bottom
        
        # Adjusted threshold for more accurate turn detection
        if abs(delta) < 25:
            return "Straight"
        elif delta < 0:
            # Determine turn severity
            if delta < -50:
                return "Sharp Left Turn"
            else:
                return "Turning Left"
        else:
            # Determine turn severity
            if delta > 50:
                return "Sharp Right Turn"
            else:
                return "Turning Right"
    except:
        # Fallback in case of calculation errors
        return "Detecting..."



video_path = "sampleVideo.mp4"  # your video path upload here
cap = cv2.VideoCapture(video_path)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("Output_Lane_det.mp4", fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    # Enhanced preprocessing pipeline for better lane detection
    
    # 1. Convert to multiple color spaces for better feature extraction
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # 2. Create masks for white and yellow lines in different color spaces
    # HLS color space for white lines
    white_hls = cv2.inRange(hls, np.array([0, 170, 0]), np.array([255, 255, 255]))
    # HSV color space for yellow lines
    yellow_hsv = cv2.inRange(hsv, np.array([15, 80, 100]), np.array([35, 255, 255]))
    # LAB color space for yellow lines (b channel is good for yellow)
    b_channel = lab[:,:,2]
    yellow_lab = cv2.inRange(b_channel, 140, 255)
    
    # 3. Combine masks
    yellow_mask = cv2.bitwise_or(yellow_hsv, yellow_lab)
    mask = cv2.bitwise_or(white_hls, yellow_mask)
    
    # 4. Apply mask to original image
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    
    # 5. Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    
    # 6. Apply histogram equalization to improve contrast
    gray = cv2.equalizeHist(gray)
    
    # 7. Apply Gaussian blur with larger kernel for better noise reduction
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # 8. Apply Canny edge detection with optimized thresholds
    edges = cv2.Canny(blur, 40, 120)
    
    # 9. Apply region of interest mask
    cropped = region_of_interest(edges)
    
    # 10. Optional: Dilate edges to connect nearby edges
    kernel = np.ones((3,3), np.uint8)
    cropped = cv2.dilate(cropped, kernel, iterations=1)

    # Advanced Hough transform with optimized parameters
    # Use a lower threshold to detect more lines
    # Use a smaller minLineLength to detect shorter line segments
    # Use a larger maxLineGap to connect nearby line segments
    lines = cv2.HoughLinesP(
        cropped,
        rho=1,              # Distance resolution in pixels
        theta=np.pi/180,    # Angle resolution in radians
        threshold=40,       # Minimum number of votes (intersections)
        minLineLength=20,   # Minimum length of line in pixels
        maxLineGap=300      # Maximum allowed gap between line segments
    )
    
    # Debug visualization of all detected lines (optional)
    # Uncomment to see all detected lines
    # line_img = np.zeros_like(frame)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line.reshape(4)
    #         cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imshow("All Lines", line_img)

    left_line, right_line = average_slope_intercept(frame, lines)


    if left_line is not None:
        left_history.append(left_line)
    if right_line is not None:
        right_history.append(right_line)


    left_avg = np.mean(left_history, axis=0).astype(int) if left_history else None
    right_avg = np.mean(right_history, axis=0).astype(int) if right_history else None

    # Create visualization layers
    overlay = draw_lane_area(frame, left_avg, right_avg)
    
    # Create a mask visualization to show the detected lane markings
    mask_viz = np.zeros_like(frame)
    mask_viz[:,:,0] = mask  # Blue channel
    mask_viz[:,:,1] = mask  # Green channel
    mask_viz[:,:,2] = mask  # Red channel
    
    # Create edge visualization
    edge_viz = np.zeros_like(frame)
    edge_viz[:,:,0] = cropped  # Blue channel
    edge_viz[:,:,1] = cropped  # Green channel
    edge_viz[:,:,2] = cropped  # Red channel
    
    # Combine visualizations
    output = cv2.addWeighted(frame, 0.8, overlay, 1, 1)
    
    # Optionally show the mask and edge visualizations
    # Uncomment these lines to see the intermediate processing steps
    # cv2.imshow("Lane Mask", mask_viz)
    # cv2.imshow("Edge Detection", edge_viz)
    
    # Add turn direction text
    direction = estimate_turn(left_avg, right_avg)
    cv2.putText(output, direction, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
    
    # Add confidence indicator based on how many frames have detected lines
    confidence = min(len(left_history) / 15.0, len(right_history) / 15.0) * 100
    confidence_text = f"Confidence: {int(confidence)}%"
    cv2.putText(output, confidence_text, (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    out.write(output)
    cv2.imshow("ADAS Lane Detection (Stable)", output)
    # Increase delay between frames to slow down playback (30ms = ~33fps)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
