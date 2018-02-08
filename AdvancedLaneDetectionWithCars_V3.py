# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 17:41:24 2018

@author: Ira
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip

import pickle
from collections import deque
from scipy.ndimage.measurements import label
from skimage.feature import hog

window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

M = np.array([[-0.50772,-1.49582,951.33],
              [-3.21965e-15,-1.98816,914.554],
              [-4.98733e-18,-0.00238604,1]
              ])

Minv = np.array([[0.192187,-0.766859,518.5],
                 [1.77636e-15,-0.502977,460],
                 [-1.73472e-18,-0.00120012,1]
                 ])

prv5L = []
prv5R = []
current_left_fit = []
current_right_fit = []
prv = 0
previous = 0
prv_left_fit = []
prv_right_fit = []
prv_left_fitx = []
prv_right_fitx = []

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Hog Feature and Visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features
    
def get_xy_steps(X,Y):
    
    window = 64
    nxblocks = (X // pix_per_cell) - 1
    nyblocks = (Y // pix_per_cell) - 1

    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    x_steps = (nxblocks - nblocks_per_window) // cells_per_step
    y_steps = (nyblocks - nblocks_per_window) // cells_per_step

    
    return x_steps, y_steps,nblocks_per_window

def get_region_hog_features(ch1, ch2, ch3):
    
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    return hog1, hog2, hog3

def extract_features(imgs, color_space='YCrCb', spatial_size=(32, 32), hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel='ALL', spatial_feat=True, hist_feat=True,
                     hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)

        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat is True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat is True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat is True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

def get_prediction(img,hog_features):
    # Get color features
    spatial_features = bin_spatial(img, size=spatial_size)
    hist_features = color_hist(img, nbins=hist_bins)
    # Scale features and make a prediction
    stacked_features = np.hstack((spatial_features, hist_features, hog_features))
    test_features = X_scaler.transform(stacked_features.reshape(1, -1))
    test_prediction = svc.predict(test_features)
    
    return test_prediction

def get_reduced_search_boundary(img):
    
    if frameCnt % 10 == 0:
        regionmsk = np.ones_like(img[:, :, 0])
    else:
        regionmsk = np.sum(np.array(heat_images), axis=0)
        regionmsk[(regionmsk > 0)] = 1
        regionmsk = cv2.dilate(regionmsk, np.ones((50, 50)), iterations=1)

    nonzero = regionmsk.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    return(nonzeroy,nonzerox)
    
def find_cars(img):

    
    global heatmap, heat_images,frameCnt,full_frame_processing_interval,xstart,ystart_ystop_scale
    global kernel,threshold 
    global X_scaler, orient, pix_per_cell,cell_per_block,spatial_size,hist_bins,svc
    
    box_list = []

    raw_frame = np.copy(img)
    img = img.astype(np.float32) / 255
    
    nonzeroy,nonzerox = get_reduced_search_boundary(img)

    frameCnt += 1

    for (ystart, ystop, scale) in ystart_ystop_scale:

        if len(nonzeroy) != 0:
            ystart = max(np.min(nonzeroy), ystart)
            ystop = min(np.max(nonzeroy), ystop)
        if len(nonzerox) != 0:
            xstart = max(np.min(nonzerox), xstart_initial)
            xstop = np.max(nonzerox)
        else:
            continue
    
        if xstop <= xstart or ystop <= ystart:
            continue
        
        img_tosearch = img[ystart:ystop, xstart:xstop, :]
        search_region = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = search_region.shape
            ys = np.int(imshape[1] / scale)
            xs = np.int(imshape[0] / scale)
            if (ys < 1 or xs < 1):
                continue
            search_region = cv2.resize(search_region, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        if search_region.shape[0] < 64 or search_region.shape[1] < 64:
            continue
        
        ch1 = search_region[:, :, 0]
        ch2 = search_region[:, :, 1]
        ch3 = search_region[:, :, 2]

        hog1,hog2,hog3 = get_region_hog_features(ch1, ch2, ch3)

        nxsteps, nysteps, nblocks_per_window = get_xy_steps(ch1.shape[1],ch1.shape[0])
#        print("nx:",nxsteps,"ny:", nysteps, "ystart:",ystart, "xstart",xstart, "ystop:",ystop, "xstop",xstop)
        
        for xb in range(nxsteps + 1):
            for yb in range(nysteps + 1):
                ypos = yb * 2
                xpos = xb * 2

                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = search_region[ytop:ytop + 64, xleft:xleft + 64]
                
                test_prediction = get_prediction(subimg,hog_features)
                if test_prediction == 1:
                    xbox_left = xstart + np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(64 * scale)
                    box_list.append(
                        ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    # Add heat to each box in box list
    heatmap, heat_images = add_heatmap_threshold(raw_frame, box_list, threshold)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_labeled_boxes(raw_frame, labels)

    return raw_frame

def add_heatmap_threshold(raw_frame, bbox_list, threshold):
    heatmap_temp = np.zeros_like(raw_frame[:, :, 0]).astype(np.float)

    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap_temp[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    heat_images.append(heatmap_temp)
    heatmap = np.sum(np.array(heat_images), axis=0)
    heatmap[heatmap <= threshold] = 0
    return heatmap, heat_images

def draw_labeled_boxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, box[0], box[1], (0, 0, 255), 6)

def direction_threshold(sobelx, sobely,  thresh=(0, np.pi/2)):

    absgraddir = np.arctan2(sobely, sobelx)
    absgraddir_degree = (absgraddir / np.pi) * 180
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir_degree >= 40) & (absgraddir_degree <= 75)] = 255

    # Return the binary image
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def magnitude_thresh(img, sobel_kernel=3, mag_thresh=(0, 255),  s_thresh=(170, 255)):
    
    # 1) Convert to grayscale
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0,sobel_kernel)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1,sobel_kernel)
    # 3) Calculate the magnitude 
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)    
    abs_sobelxy= np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))    
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    sxbinary_x = np.zeros_like(scaled_sobelx)
    sxbinary_y = np.zeros_like(scaled_sobely)    
    sxbinary_x[(scaled_sobelx >= mag_thresh[0]) & (scaled_sobelx <= mag_thresh[1])] = 255
    sxbinary_y[(scaled_sobely >= mag_thresh[0]) & (scaled_sobely <= mag_thresh[1])] = 255
    
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 255

    return sxbinary, abs_sobelx, abs_sobely

def getCurvature(ploty,left_fit,right_fit,leftx,rightx):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    #print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / \
    np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / \
    np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    return(left_curverad,right_curverad)
    # Example values: 632.1 m    626.2 m

   
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def calibrate_camera(Image_Path):
    global counter
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objpoints = []
    imgpoints = []
    
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    images = glob.glob(Image_Path) #
    
    counter = 0
    
    for fname in images:
       img = cv2.imread(fname)
       gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
       # Find the chess board corners
       ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
       
       # If found, add object points, image points (after refining them)
       if ret == True:
           objpoints.append(objp)
        
           cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
           imgpoints.append(corners)
        
           # Draw and display the corners
           cv2.drawChessboardCorners(img, (9,6), corners,ret)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints, gray.shape[::-1],None, None)
    return ret, mtx, dist, rvecs, tvecs

def perspectiveTransform(img):

    src_vertices = np.array([[(587, 446), (153, 673), (1126, 673), (691, 446)]],dtype=np.float32) 
    dst_vertices = np.array([[(200, 0), (200, 720), (1080, 720), (1080, 0)]],dtype=np.float32)   

    M = cv2.getPerspectiveTransform(src_vertices, dst_vertices)
    Minv = cv2.getPerspectiveTransform(dst_vertices, src_vertices)

    return(M,Minv)
    
def hls_mask(img):

    white_lwr = np.array([0, 210, 0])
    white_upr = np.array([255, 255, 255])
    
    yellow_lwr = np.array([18, 0, 100])
    yellow_upr = np.array([30, 220, 255])
    
    # Convert the scale from RGB to HLS
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # white color mask
    white_mask = cv2.inRange(hls_img, white_lwr, white_upr)


    # yellow color mask
    yellow_mask = cv2.inRange(hls_img, yellow_lwr, yellow_upr)

    return white_mask, yellow_mask


def processFrame(image):
    global start
    global prv_left_fit 
    global prv_right_fit
    global prv_curvature
 
    y_eval = 700 #np.max(ploty)
    midx = 640
    xm_per_pix = 3.7/660.0 # meters per pixel in x dimension
    ym_per_pix = 30/720 # meters per pixel in y dimension
    nwindows = 9
    margin = 100
    minpix = 50

    #undistort the image
    dst = cv2.undistort(image,mtx, dist, None, mtx) 

    #find the magnitude of the gradient 
    mag_binary, sobel_absX, sobel_absY = magnitude_thresh(dst, \
                                                          sobel_kernel=3, \
                                                          mag_thresh=(30, 150), \
                                                          s_thresh=(170, 255))
    
    #find the direction of the gradient
    dir_binary = direction_threshold(sobel_absX,sobel_absY,thresh=(0.7,1.3))

    combined_MagDir = np.zeros_like(mag_binary)
    combined_MagDir[((mag_binary == 255) & (dir_binary == 255))] = 255  
     
    w_color, y_color = hls_mask(dst)
    
    combined = np.zeros_like(w_color)
    combined[((w_color == 255) | (y_color == 255))] = 255
    combined[(combined == 255)] = 255

#    temp = np.zeros_like(w_color)
#    temp[((combined == 255)|(combined_MagDir== 255))] = 255
#    
#    combined = temp
    
    warped = cv2.warpPerspective(combined, M, (1280, 720),flags=cv2.INTER_LINEAR)
    
    window_height = np.int(warped.shape[0]/nwindows)
    
    if start:
        histogram = np.sum(warped[int(warped.shape[0]/2):,:], axis=0)

        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            win_y_low = warped.shape[0] - (window+1)*window_height
            win_y_high = warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & \
                              (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & \
                               (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        prv_right_fit = right_fit
        prv_left_fit = left_fit
        
        y1 = (2*right_fit[0]*y_eval + right_fit[1])*xm_per_pix/ym_per_pix
        y2 = 2*right_fit[0]*xm_per_pix/(ym_per_pix**2)
        curvature = ((1 + y1*y1)**(1.5))/np.absolute(y2)
        
        if (curvature) < 500:
            prv_curvature = 0.75*curvature + 0.25*(((1 + y1*y1)**(1.5))/np.absolute(y2)) 
        
        start = 0
 
    else:
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_fit = prv_left_fit
        right_fit = prv_right_fit
        
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & \
                          (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & \
                           (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        left_error = ((prv_left_fit[0] - left_fit[0]) ** 2).mean(axis=None)      
        right_error = ((prv_right_fit[0] - right_fit[0]) ** 2).mean(axis=None)        
        if left_error < 0.01:
            prv_left_fit = 0.75 * prv_left_fit + 0.25 * left_fit   
        if right_error < 0.01:
            prv_right_fit = 0.75 * prv_right_fit + 0.25 * right_fit
        
        y1 = (2*right_fit[0]*y_eval + right_fit[1])*xm_per_pix/ym_per_pix
        y2 = 2*right_fit[0]*xm_per_pix/(ym_per_pix**2)
        curvature = ((1 + y1*y1)**(1.5))/np.absolute(y2)

        prv_curvature = curvature
              
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    
    cv2.putText(result,'Radius of Curvature: %.2fm' % curvature,(20,40), \
                cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    
    x_left_pix = left_fit[0]*(y_eval**2) + left_fit[1]*y_eval + left_fit[2]
    x_right_pix = right_fit[0]*(y_eval**2) + right_fit[1]*y_eval + right_fit[2]
        
    position_from_center = ((x_left_pix + x_right_pix)/2.0 - midx) * xm_per_pix

    if position_from_center < 0:
        text = 'left'
    else:
        text = 'right'
    cv2.putText(result,'Distance From Center: %.2fm %s' % (np.absolute(position_from_center), text),(20,80), \
                cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    withCarsResult = find_cars(result)
    
    return withCarsResult
    


#calibrate the camera
ret, mtx, dist, rvecs, tvecs = calibrate_camera('.\\camera_cal\\*.jpg')

frameCnt = 0
start = 1
prv_left_fit = [np.array([False])] 
prv_right_fit = [np.array([False])] 
prv_curvature = 0
# Current HeatMap
heatmap = None
# Heat Image for the Last Three Frames
heat_images = deque(maxlen=3)
xstart_initial = 600
ystart_ystop_scale = [(380, 480, 1), (400, 600, 1.5), (500, 700, 2.5)]
threshold = 3


# Loading Model Parameters
with open('model-params.pk', 'rb') as pfile:
    pickle_data = pickle.load(pfile)
#    for key in pickle_data:
    svc = pickle_data['svc']
    X_scaler  = pickle_data['X_scaler']
    color_space  = pickle_data['color_space']
    orient  = pickle_data['orient']
    pix_per_cell  = pickle_data['pix_per_cell']
    cell_per_block  = pickle_data['cell_per_block']
    spatial_size  = pickle_data['spatial_size']
    hist_bins  = pickle_data['hist_bins']
    hog_channel  = pickle_data['hog_channel']
    del pickle_data

#test_images = glob.glob('.\\test_Images\\*.jpg')
#for fname in test_images:
#    img = mpimg.imread(fname)   
#    temp = fname.split('\\')
#    filename = temp[2].split('.jpg')
#    temp1 ='.\\test_Images\\'+ filename[0]+'_out.jpg'
#    result = find_cars(img)
##    lab = hls_mask(img)
#    cv2.imwrite(temp1,result)

##load the video and process frame by frame
#undist_output = 'test_video_out.mp4'
#clip2 = VideoFileClip('test_video.mp4')
#yellow_clip = clip2.fl_image(processFrame, apply_to=[])
#yellow_clip.write_videofile(undist_output, audio=False)

# load the video and process frame by frame
undist_output = 'project_video_out.mp4'
clip2 = VideoFileClip('project_video.mp4')
yellow_clip = clip2.fl_image(processFrame, apply_to=[])
yellow_clip.write_videofile(undist_output, audio=False)
