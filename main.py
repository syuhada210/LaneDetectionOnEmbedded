import cv2
import numpy as np
import time

from adas_helper_functions import *
from ldw_helper_functions import *

countL = 0
countR = 0
c = 0
number = 0
counter = 0

ref_left = np.array([-0.0001, 0, 300])
ref_right = np.array([-0.0001, 0, 500])
left_fit = np.array([-0.0001, 0, 300])
right_fit = np.array([-0.0001, 0, 500])

currentL = np.zeros([3])
previousL = np.zeros([3])
currentR = np.zeros([3])
previousR = np.zeros([3])

DAY_BRIGHTNESS_TRESH = 30


def run_pipeline(frame):
    global ref_left, ref_right

    # Pre-processing

    """STEP#0 SCALE TO FIXED"""
    frame = scale_to_fixed(frame)
    # (h2, w2) = frame.shape[:2]

    brightness = getBrightness(frame)

    # ---------LDW PIPELINE

    """LDW-STEP#1 CROP ROAD ROI"""
    crop_road_roi_frame = crop_road_roi(frame, frame_output=False)

    """LDW-STEP#2 PERSPECTIVE TRANSFORM"""
    warped_image = apply_perspective_transform(frame, crop_road_roi_frame, toWarp=True)
    cv2.imshow("warp",warped_image)
    blur = cv2.GaussianBlur(warped_image, (5, 5), 0)
    """LDW-STEP#3 TIME CONDITION CHECK & BINARIZE"""
    if brightness > DAY_BRIGHTNESS_TRESH:
        # WHEN CLEAR BRIGHT DAY
        sharpened_image = sharpened(blur)
        binaryImage = compute_binary_image(warped_image, window_name="1st binary", output_show=False)
        # cv2.imshow("binary",binaryImage)
        whites = cv2.countNonZero(binaryImage)
        # print(whites)
        # WHITE COUNTS FOR WHEN IT RAINS
        if 0 < whites < 3000:
            binaryImage = edge_filter(warped_image, binaryImage)

        else:
            # morph top - remove big region
            # morph open - remove small region ie. erosion
            kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            environment = cv2.morphologyEx(binaryImage, cv2.MORPH_TOPHAT, kernel_1)
            kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binaryImage = cv2.morphologyEx(environment, cv2.MORPH_OPEN, kernel_2)

    else:
        # WHEN IT IS NOT CLEAR BRIGHT DAY (NIGHT || UNDER BRIDGE)
        # denoise = cv2.fastNlMeansDenoisingColored(warped_image)
        sharpened_image = sharpened(warped_image)
        # sharpened_image = cv2.bilateralFilter(sharpened_image, 9, 18, 9 / 2)
        # cv2.imshow("bila",sharpened_image)
        binaryImage1 = compute_binary_image(sharpened_image, window_name="1st binary", output_show=False)
        binaryImage = edge_filter(warped_image, binaryImage1)

        # whites = cv2.countNonZero(binaryImage)

    """LDW-STEP#4 DEFINE LANE-LINES"""
    leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds = extract_lanes_pixels(binaryImage, plot_show=True)
    left_fit, right_fit = poly_fit(leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds,
                                   binaryImage, output_show=False)

    """LDW-STEP#5 SANITY CHECK"""
    sanity_stat = sanity_check(leftx, rightx, left_fit, right_fit, 0, .25)

    """LDW-STEP#6 DEAL WITH RELIABLE & UNRELIABLE FIT"""
    if sanity_stat:
        # Save as last known reliable fit
        ref_left, ref_right = left_fit, right_fit
        left_fit_up, right_fit_up = left_fit, right_fit
    else:
        if len(leftx) == 0:
            right = right_fit[0] * 540 ** 2 + right_fit[1] * 540 + right_fit[2]
            left_fit_up = np.array([-0.0001, 0, right - 200])
            right_fit_up = right_fit

        elif len(rightx) == 0:
            left = left_fit[0] * 540 ** 2 + left_fit[1] * 540 + left_fit[2]
            right_fit_up = np.array([-0.0001, 0, left + 200])
            left_fit_up = left_fit

        else:
            left_fit_up, right_fit_up = ref_left, ref_right

    """LDW-STEP#7 TRIGGER WARNING"""
    color, center = warning(left_fit_up, right_fit_up)

    if color == 1:
        color = (0, 0, 255)
    elif color == 2:
        color = (0, 255, 255)
    else:
        color = (255, 215, 0)

    warped_color_img = drawLine(frame, binaryImage, left_fit_up, right_fit_up, color, int(center))

    dewarped_image = apply_perspective_transform(frame, warped_color_img,  toWarp=False)

    frame = cv2.addWeighted(dewarped_image, 0.8, frame, 1.0, 0)
    cv2.line(frame, (int(640 / 2), 360), (int(640 / 2), 0), (0, 0, 255), 1)

    return frame


# source_frame = cv2.VideoCapture("C:/Users/default.DESKTOP-O651TTO/Documents/LDWS/SiangHW.mp4") #Day time video
# source_frame = cv2.VideoCapture("C:/Users/default.DESKTOP-O651TTO/Documents/Video ADAS/WIN_20180803_205421.MP4") # Night time video
# source_frame = cv2.VideoCapture("C:/Users/default.DESKTOP-O651TTO/Documents/Video ADAS/WIN_20180821_114234.MP4")
source_frame = cv2.VideoCapture("C:\\Users\\Sayang Nurul\\Lane detection2\\640 x 360\\Straight Lane.mp4")#Siang
 #  source_frame = cv2.VideoCapture("C:\\Users\\lenovo\\Documents\\Video ADAS\\WIN_20180803_205421.MP4") #mlm

_, test_frame = source_frame.read()
(h, w) = test_frame.shape[:2]
createtrackbar(h, w)
start = time.time()
count=0
while True:
    _, frame = source_frame.read()

    final_merge = run_pipeline(frame)

    cv2.imshow("FINAL-OUTPUT", final_merge)
    count=count+1;
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

end = time.time()
duration = end-start
print("time taken: ",duration, count)
source_frame.release()
cv2.destroyAllWindows()
