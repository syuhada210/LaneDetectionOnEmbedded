import os, glob, pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.ion()

fixed_scaled_frame_width = 0
fixed_scaled_frame_height = 0

crop_height = 0
number = 0

transformation_matrix = np.float32([[1, 0, 0], [0, 1, crop_height]])


def createtrackbar(h, w):
    cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)

    # ldw
    cv2.createTrackbar('r', 'Calibration', 0, int((w / 4) - 1), getvalue)  # warping
    cv2.createTrackbar('l', 'Calibration', 0, int((w / 4) - 1), getvalue)

    cv2.setTrackbarPos('r', 'Calibration', 132)
    cv2.setTrackbarPos('l', 'Calibration', 132)

    cv2.createTrackbar('croph', 'Calibration', 1, int((1080 / 2) - 1), getvalue)
    cv2.setTrackbarPos('croph', 'Calibration', 218)  # 365 utk video webcam,

    cv2.createTrackbar('cropw', 'Calibration', 1, int((1920 / 4) - 1), getvalue)
    cv2.setTrackbarPos('cropw', 'Calibration', 200)

    cv2.createTrackbar('LS1', 'Calibration', 0, 255, getvalue)
    cv2.createTrackbar('US1', 'Calibration', 0, 255, getvalue)
    cv2.createTrackbar('LS2', 'Calibration', 0, 255, getvalue)
    cv2.createTrackbar('US2', 'Calibration', 0, 255, getvalue)
    cv2.createTrackbar('LV1', 'Calibration', 0, 255, getvalue)
    cv2.createTrackbar('UV1', 'Calibration', 0, 255, getvalue)
    cv2.createTrackbar('LV2', 'Calibration', 0, 255, getvalue)
    cv2.createTrackbar('UV2', 'Calibration', 0, 255, getvalue)
    cv2.setTrackbarPos('LS1', 'Calibration', 0)
    cv2.setTrackbarPos('US1', 'Calibration', 255)
    cv2.setTrackbarPos('LS2', 'Calibration', 0)
    cv2.setTrackbarPos('US2', 'Calibration', 255)
    cv2.setTrackbarPos('LV1', 'Calibration', 200)
    cv2.setTrackbarPos('UV1', 'Calibration', 255)
    cv2.setTrackbarPos('LV2', 'Calibration', 140)
    cv2.setTrackbarPos('UV2', 'Calibration', 255)


def getvalue(value):
    return value


def scale_to_fixed(frame):
    # Scale incoming image to 540x960
    global fixed_scaled_frame_height, fixed_scaled_frame_width

    scalewidth = frame.shape[1] / 1280
    scaleheight = frame.shape[0] / 720

    frame = cv2.resize(frame, (0, 0), fx=1 / 2 / scaleheight, fy=1 / 2 / scalewidth)
    (fixed_scaled_frame_height, fixed_scaled_frame_width) = frame.shape[:2]

    return frame


def getBrightness(frame):
    b, g, r = cv2.split(frame)
    b = b * 0.114
    g = g * 0.587
    r = r * 0.299
    lum = b + g + r
    m = cv2.mean(lum)
    brightness = m[0]

    brightness = brightness / 255 * 100
    return brightness


def CropFirstROI(frame):
    (h, w) = frame.shape[:2]

    cropping_height = cv2.getTrackbarPos('croph', 'Calibration')

    frameROI = frame[cropping_height:int(h * 0.9), :]

    return frameROI, cropping_height


def compute_perspective_transform(frame, toEagleEye=True):
    # Define 4 source and 4 destination points = np.float32([[,],[,],[,],[,]])
    eagleEyeRightSide = cv2.getTrackbarPos('r', 'Calibration')
    eagleEyeLeftSide = cv2.getTrackbarPos('l', 'Calibration')
    cropping_h = cv2.getTrackbarPos('croph', 'Calibration')

    x1 = 0
    y1 = cropping_h
    x2 = frame.shape[1] - 1
    y2 = cropping_h
    x3 = 0
    y3 = frame.shape[0] * 0.9 - 1
    x4 = frame.shape[1] - 1
    y4 = frame.shape[0] * 0.9 - 1

    src = np.array([(x1, y1), (x2, y2), (x4, y4), (x3, y3)], dtype="float32")
    W = frame.shape[1]
    L = frame.shape[0]

    dst = np.array([(0, 0), (W - 1, 0), (W / 2 + eagleEyeRightSide, L - 1), (W / 2 - eagleEyeLeftSide, L - 1)],
                   dtype="float32")
    if toEagleEye is True:
        M = cv2.getPerspectiveTransform(src, dst)
    elif toEagleEye is False:
        M = cv2.getPerspectiveTransform(dst, src)

    return M


def apply_perspective_transform(frame, frame2, toWarp=True):
    global transformation_matrix
    crop_w_value = cv2.getTrackbarPos('cropw', 'Calibration')

    if toWarp is True:
        transformation_matrix = compute_perspective_transform(frame2, toEagleEye=True)
        warped_image = cv2.warpPerspective(frame2, transformation_matrix, (frame2.shape[1], frame2.shape[0]),
                                           flags=cv2.INTER_NEAREST)  # keep same size as input image
        # temp = np.zeros_like(frame)
        # temp[:, crop_w_value:-crop_w_value] = warped_image[:, crop_w_value:-crop_w_value]
        temp = warped_image
    elif toWarp is False:
        transformation_matrix = G = compute_perspective_transform(frame2, False)
        warped_image = cv2.warpPerspective(frame2, G, (frame2.shape[1], frame2.shape[0]),
                                           flags=cv2.INTER_NEAREST)  # keep same size as input image
        temp = warped_image

    return temp  # warped_image


def sharpened(warped_image):
    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])

    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(warped_image, -1, kernel_sharpening)


    return sharpened


def compute_binary_image(color_image, window_name="binary", output_show=False):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)

    LS1 = cv2.getTrackbarPos('LS1', 'Calibration')
    US1 = cv2.getTrackbarPos('US1', 'Calibration')
    LS2 = cv2.getTrackbarPos('LS2', 'Calibration')
    US2 = cv2.getTrackbarPos('US2', 'Calibration')
    LV1 = cv2.getTrackbarPos('LV1', 'Calibration')
    UV1 = cv2.getTrackbarPos('UV1', 'Calibration')
    LV2 = cv2.getTrackbarPos('LV2', 'Calibration')
    UV2 = cv2.getTrackbarPos('UV2', 'Calibration')

    boundaries = ([0, LS1, LV1], [179, US1, UV1])
    lower = np.array(boundaries[0], dtype=np.uint8)
    upper = np.array(boundaries[1], dtype=np.uint8)
    WnB1 = cv2.inRange(hsv, lower, upper)

    boundaries = ([80, LS2, LV2], [179, US2, UV2])
    lower = np.array(boundaries[0], dtype=np.uint8)
    upper = np.array(boundaries[1], dtype=np.uint8)
    WnB2 = cv2.inRange(hsv, lower, upper)

    boundaries = ([0, 73, 0], [100, 255, 255])
    lower = np.array(boundaries[0], dtype=np.uint8)
    upper = np.array(boundaries[1], dtype=np.uint8)
    WnB3 = cv2.inRange(hsv, lower, upper)

    combined_w_Y = WnB1 | WnB2 | WnB3

    if output_show:
        cv2.imshow(window_name, combined_w_Y)

    return combined_w_Y


def edge_filter(cropped_image, binary_frame):
    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 50
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    # combined_binary[(s_binary == 255) & (sxbinary == 255)] = 255
    combined_binary = cv2.bitwise_or(binary_frame, sxbinary, combined_binary)

    return combined_binary


def extract_lanes_pixels(binary_warped, plot_show=False):
    # Set the width of the windows +/- margin
    margin = 20

    # Set minimum number of pixels found to recenter window
    minpix = 100

    # Choose the number of sliding windows
    nwindows = 20

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)

    if plot_show:
        plt.plot(histogram)
        plt.pause(0.0001)
        plt.show()
        plt.gcf().clear()

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Create 3 channels to draw green rectangle
    out_img = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # distance = win_xright_high - win_xleft_high
        # print("dist", distance) #140 is good distance

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        # https://stackoverflow.com/questions/7924033/understanding-numpys-nonzero-function
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        cv2.imshow("window detect", out_img)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds


def poly_fit(leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, binary_warped, output_show=False):
    # Fit a second order polynomial to each
    global left_fit, right_fit
    h = 360
    try:
        if len(leftx) != 0 and len(rightx) != 0:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        elif len(leftx) == 0:
            right_fit = np.polyfit(righty, rightx, 2)
            right = right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2]
            left_fit = np.array([-0.0001, 0, right - 200])
        elif len(rightx) == 0:
            left_fit = np.polyfit(lefty, leftx, 2)
            left = left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2]
            right_fit = np.array([-0.0001, 0, left + 200])

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if output_show:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            cv2.imshow("poly_fit)output", out_img)

    except Exception as e:
        print(e)
        # pass

    return left_fit, right_fit


def sanity_check(leftx, rightx, left_fit, right_fit, minSlope, maxSlope):
    # Performs a sanity check on the lanes
    # Check 1: check if left and right fits exists
    # Check 2: Calculates the tangent between left and right in two points, and check if it is in a reasonable threshold
    h = 360
    if len(leftx) == 0 or len(rightx) == 0:
        status = False
        # Previous fitlines routine returns empty list to them if not finds
    else:
        # Difference of slope, slope = 2ax+b
        slopeL = 2 * left_fit[0] * h + left_fit[1]
        slopeR = 2 * right_fit[0] * h + right_fit[1]
        delta_slope = np.abs(slopeL - slopeR)

        if minSlope <= delta_slope <= maxSlope:
            # h = 360
            yl = left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2]
            # print("yl", yl)
            yr = right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2]
            # print("yr", yr)
            diffLeftRight = yr - yl

            if 180 < diffLeftRight < 300:
                status = True
            else:
                status = False
        else:
            status = False

    return status


def warning(left_fit, right_fit):
    global number
    h2 = 360
    yl = left_fit[0] * h2 ** 2 + left_fit[1] * abs(h2) + left_fit[2]
    # print("yl", yl)
    yr = right_fit[0] * h2 ** 2 + right_fit[1] * abs(h2) + right_fit[2]
    # print("yr", yr)

    centerlane = (yl + yr) / 2
    midframe = 640 / 2

    diffLeftRight = yr - yl

    if 150 < diffLeftRight < 250:

        diffleft = midframe - yl
        diffright = yr - midframe

        diff = abs(diffright - diffleft)

        if diff >= 100 and diff <=150:
            number = 1
        elif 50 < diff < 100:
            number = 2
        else:
            number = 0

    else:
        print("error")
        number = 0

    return number, centerlane


def drawLine(undist, warped, left_fit, right_fit, color, center):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    # Fit new polynomials to x,y in world space
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # print(left_fitx)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # print(np.int_(pts))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), color)
    # cv2.line(undist, (540, center), (400,center), (255, 0, 0), 2)
    cv2.line(color_warp, (center, 360), (center, int(360 * 2 / 3)), (255, 0, 0), 5)
    # (255, 215, 0)

    return color_warp


####################################################################

def crop_road_roi(frame, frame_output=False):
    global crop_height, transformation_matrix
    first_roi, crop_height = CropFirstROI(frame)
    transformation_matrix = np.float32([[1, 0, 0], [0, 1, crop_height]])
    dst = cv2.warpAffine(first_roi, transformation_matrix,
                         (int(fixed_scaled_frame_width), int(fixed_scaled_frame_height)))  # translation

    if frame_output:
        cv2.imshow("frameroi", dst)

    return frame

