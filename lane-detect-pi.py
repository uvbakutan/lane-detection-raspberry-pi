import picamera
from picamera.array import PiRGBArray
import numpy as np
import cv2
import time
import warnings

warnings.filterwarnings('error')

image_size=(320, 192)
camera = picamera.PiCamera()
camera.resolution = image_size
camera.framerate = 7 
camera.vflip = False
camera.hflip = False 
#camera.exposure_mode='off'
rawCapture = PiRGBArray(camera, size=image_size)

# allow the camera to warmup
time.sleep(0.1)

# class for lane detection
class Lines():
    def __init__(self):
        # were the lines detected at least once
        self.detected_first = False
        # were the lines detected in the last iteration?
        self.detected = False
        # average x values of the fitted lines
        self.bestxl = None
        self.bestyl = None
        self.bestxr = None
        self.bestyr = None
        # polynomial coefficients averaged over the last iterations
        self.best_fit_l = None
        self.best_fit_r = None
        #polynomial coefficients for the most recent fit
        self.current_fit_l = None
        self.current_fit_r = None
        # radius of curvature of the lines in meters
        self.left_curverad = None
        self.right_curverad = None
        #distance in meters of vehicle center from the line
        self.offset = None
        # x values for detected line pixels
        self.allxl = None
        self.allxr = None
        # y values for detected line pixels
        self.allyl = None
        self.allyr = None
        # camera calibration parameters
        self.cam_mtx = None
        self.cam_dst = None
        # camera distortion parameters
        self.M = None
        self.Minv = None
        # image shape
        self.im_shape = (None,None)
        # distance to look ahead in meters
        self.look_ahead = 10
        self.remove_pixels = 90
        # enlarge output image
        self.enlarge = 2.5
        # warning from numpy polyfit
        self.poly_warning = False

    # set camera calibration parameters
    def set_cam_calib_param(self, mtx, dst):
        self.cam_mtx = mtx
        self.cam_dst = dst

    # undistort image
    def undistort(self, img):
        return cv2.undistort(img, self.cam_mtx, self.cam_dst, None,self.cam_mtx)

    # get binary image based on color thresholding
    def color_thresh(self, img, thresh=(0, 255)):
        # convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        s_channel = hsv[:,:,2]

        # threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
        return s_binary

    # get binary image based on sobel gradient thresholding
    def abs_sobel_thresh(self, sobel, thresh=(0, 255)):

        abs_sobel = np.absolute(sobel)

        max_s = np.max(abs_sobel)
        if max_s == 0:
            max_s=1

        scaled_sobel = np.uint8(255*abs_sobel/max_s)

        sbinary = np.zeros_like(scaled_sobel)
        sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return sbinary

    # get binary image based on sobel magnitude gradient thresholding
    def mag_thresh(self, sobelx, sobely, mag_thresh=(0, 255)):

        abs_sobel = np.sqrt(sobelx**2 + sobely**2)

        max_s = np.max(abs_sobel)
        if max_s == 0:
            max_s=1

        scaled_sobel = np.uint8(255*abs_sobel/max_s)

        sbinary = np.zeros_like(scaled_sobel)
        sbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

        return sbinary

    # get binary image based on directional gradient thresholding
    def dir_threshold(self, sobelx, sobely, thresh=(0, np.pi/2)):

        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)
        grad_sobel = np.arctan2(abs_sobely, abs_sobelx)

        sbinary = np.zeros_like(grad_sobel)
        sbinary[(grad_sobel >= thresh[0]) & (grad_sobel <= thresh[1])] = 1

        return sbinary

    # get binary combining various thresholding methods
    def binary_extraction(self,image, ksize=3):
        # undistort first
        #image = self.undistort(image)

        color_bin = self.color_thresh(image,thresh=(90, 150))              # initial values 110, 255

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize)

        gradx = self.abs_sobel_thresh(sobelx, thresh=(100, 190))             # initial values 40, 160
        grady = self.abs_sobel_thresh(sobely, thresh=(100, 190))             # initial values 40, 160
        mag_binary = self.mag_thresh(sobelx, sobely, mag_thresh=(100, 190))  # initial values 40, 160
        #dir_binary = self.dir_threshold(sobelx, sobely, thresh=(0.7, 1.3))

        combined = np.zeros_like(gradx)
        #combined[(((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))) | (color_bin==1) ] = 1
        combined[(((gradx == 1) & (grady == 1)) | (mag_binary == 1)) | (color_bin==1) ] = 1
        #combined[(((gradx == 1) & (grady == 1)) | (mag_binary == 1)) ] = 1

        return combined

    # transform perspective
    def trans_per(self, image):

        image = self.binary_extraction(image)

        self.binary_image = image

        ysize = image.shape[0]
        xsize = image.shape[1]

        # define region of interest
        left_bottom = (xsize/10, ysize)
        apex_l = (xsize/2 - 2600/(self.look_ahead**2),  ysize - self.look_ahead*275/30)
        apex_r = (xsize/2 + 2600/(self.look_ahead**2),  ysize - self.look_ahead*275/30)
        right_bottom = (xsize - xsize/10, ysize)

        # define vertices for perspective transformation
        src = np.array([[left_bottom], [apex_l], [apex_r], [right_bottom]], dtype=np.float32)
        dst = np.float32([[xsize/3,ysize],[xsize/4.5,0],[xsize-xsize/4.5,0],[xsize-xsize/3, ysize]])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

        if len(image.shape) > 2:
            warped = cv2.warpPerspective(image, self.M, image.shape[-2:None:-1], flags=cv2.INTER_LINEAR)
        else:
            warped = cv2.warpPerspective(image, self.M, image.shape[-1:None:-1], flags=cv2.INTER_LINEAR)
        return warped

    # creat window mask for lane detecion
    def window_mask(self, width, height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), \
               max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output

    # find widow centroids of left and right lane
    def find_window_centroids(self, warped, window_width, window_height, margin):

        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)

        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(warped.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))

        return window_centroids

    # fit polynomials on the extracted left and right lane
    def get_fit(self, image):

        # check if the lanes were detected in the last iteration, if not search for the lanes
        if not self.detected:
            # window settings
            window_width = 40
            window_height = 40 # break image into 9 vertical layers since image height is 720
            margin = 10 # how much to slide left and right for searching

            window_centroids = self.find_window_centroids(image, window_width, window_height, margin)

            # if we found any window centers
            if len(window_centroids) > 0:

                # points used to draw all the left and right windows
                l_points = np.zeros_like(image)
                r_points = np.zeros_like(image)

                # go through each level and draw the windows
                for level in range(0,len(window_centroids)):
                    # Window_mask is a function to draw window areas
                    l_mask = self.window_mask(window_width,window_height,image,window_centroids[level][0],level)
                    r_mask = self.window_mask(window_width,window_height,image,window_centroids[level][1],level)
                    # Add graphic points from window mask here to total pixels found
                    l_points[(image == 1) & (l_mask == 1) ] = 1
                    r_points[(image == 1) & (r_mask == 1) ] = 1

                # construct images of the results
                template_l = np.array(l_points*255,np.uint8) # add left window pixels
                template_r = np.array(r_points*255,np.uint8) # add right window pixels
                zero_channel = np.zeros_like(template_l) # create a zero color channel
                left_right = np.array(cv2.merge((template_l,zero_channel,template_r)),np.uint8) # make color image left and right lane

                # get points for polynomial fit
                self.allyl,self.allxl = l_points.nonzero()
                self.allyr,self.allxr = r_points.nonzero()


                # check if lanes are detected
                if (len(self.allxl)>0) & (len(self.allxr)>0):
                    try:
                        self.current_fit_l = np.polyfit(self.allyl,self.allxl, 2)
                        self.current_fit_r = np.polyfit(self.allyr,self.allxr, 2)
                        self.poly_warning = False
                    except np.RankWarning:
                        self.poly_warning = True
                        pass

                    # check if lanes are detected correctly
                    if self.check_fit():
                        self.detected = True

                        # if this is the first detection initialize the best values
                        if not self.detected_first:
                            self.best_fit_l = self.current_fit_l
                            self.best_fit_r = self.current_fit_r
                        # if not then average with new
                        else:
                            self.best_fit_l = self.best_fit_l*0.6 + self.current_fit_l * 0.4
                            self.best_fit_r = self.best_fit_r*0.6 + self.current_fit_r * 0.4

                        # assign new best values based on this iteration
                        self.detected_first = True
                        self.bestxl = self.allxl
                        self.bestyl = self.allyl
                        self.bestxr = self.allxr
                        self.bestyr = self.allyr
                        self.left_right = left_right

                    # set flag if lanes are not detected correctly
                    else:
                        self.detected = False

        # if lanes were detected in the last frame, search area for current frame
        else:
            non_zero_y, non_zero_x = image.nonzero()

            margin = 10 # search area margin
            left_lane_points_indx = ((non_zero_x > (self.best_fit_l[0]*(non_zero_y**2) + self.best_fit_l[1]*non_zero_y + self.best_fit_l[2] - margin)) & (non_zero_x < (self.best_fit_l[0] *(non_zero_y**2) + self.best_fit_l[1]*non_zero_y + self.best_fit_l[2] + margin)))
            right_lane_points_indx = ((non_zero_x > (self.best_fit_r[0]*(non_zero_y**2) + self.best_fit_r[1]*non_zero_y + self.best_fit_r[2] - margin)) & (non_zero_x < (self.best_fit_r[0]*(non_zero_y**2) + self.best_fit_r[1]*non_zero_y + self.best_fit_r[2] + margin)))

            # extracted lef lane pixels
            self.allxl= non_zero_x[left_lane_points_indx]
            self.allyl= non_zero_y[left_lane_points_indx]

            # extracted rightt lane pixels
            self.allxr= non_zero_x[right_lane_points_indx]
            self.allyr= non_zero_y[right_lane_points_indx]

            # if lines were found
            if (len(self.allxl)>0) & (len(self.allxr)>0):
                try:
                    self.current_fit_l = np.polyfit(self.allyl,self.allxl, 2)
                    self.current_fit_r = np.polyfit(self.allyr,self.allxr, 2)
                except np.RankWarning:
                    self.poly_warning = True
                    pass

                # check if lanes are detected correctly
                if self.check_fit():
                    # average out the best fit with new values
                    self.best_fit_l = self.best_fit_l*0.6 + self.current_fit_l * 0.4
                    self.best_fit_r = self.best_fit_r*0.6 + self.current_fit_r * 0.4

                    # assign new best values based on this iteration
                    self.bestxl = self.allxl
                    self.bestyl = self.allyl
                    self.bestxr = self.allxr
                    self.bestyr = self.allyr

                    # construct images of the results
                    template_l = np.copy(image).astype(np.uint8)
                    template_r = np.copy(image).astype(np.uint8)

                    template_l[non_zero_y[left_lane_points_indx],non_zero_x[left_lane_points_indx]] = 255 # add left window pixels
                    template_r[non_zero_y[right_lane_points_indx],non_zero_x[right_lane_points_indx]] = 255 # add right window pixels
                    zero_channel = np.zeros_like(template_l) # create a zero color channel
                    self.left_right = np.array(cv2.merge((template_l,zero_channel,template_r)),np.uint8) # make color image left and right lane

                # set flag if lanes are not detected correctly
                else:
                    self.detected = False

    # check if lanes are detected correctly
    def check_fit(self):
        # Generate x and y values of the fit
        ploty = np.linspace(0, self.im_shape[0]-1, self.im_shape[0])
        left_fitx = self.current_fit_l[0]*ploty**2 + self.current_fit_l[1]*ploty + self.current_fit_l[2]
        right_fitx = self.current_fit_r[0]*ploty**2 + self.current_fit_r[1]*ploty + self.current_fit_r[2]

        # find max, min and mean distance between the lanes
        max_dist  = np.amax(np.abs(right_fitx - left_fitx))
        min_dist  = np.amin(np.abs(right_fitx - left_fitx))
        mean_dist = np.mean(np.abs(right_fitx - left_fitx))
        # check if the lanes don't have a big deviation from the mean
        if (max_dist > 250) |  (np.abs(max_dist - mean_dist)> 100) | (np.abs(mean_dist - min_dist) > 100) | (mean_dist<50) | self.poly_warning:
            return False
        else:
            return True

    def calculate_curvature_offset(self):

        if self.detected_first:
            # define y value near the car
            y_eval = self.im_shape[0]

            # define conversions in x and y from pixels space to meters
            ym_per_pix = 50/250 # meters per pixel in y dimension
            xm_per_pix = 3.7/75 # meters per pixel in x dimension

            # create new polynomials to x,y in world space
            try:
                left_fit_cr = np.polyfit(self.bestyl*ym_per_pix, self.bestxl*xm_per_pix, 2)
                right_fit_cr = np.polyfit(self.bestyr*ym_per_pix, self.bestxr*xm_per_pix, 2)
            except np.RankWarning:
                 self.poly_warning = True
                 pass

            # if the poly fit is ok proceed
            if not self.poly_warning:
                # calculate the new radii of curvature
                left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
                right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
                # now our radius of curvature is in meters

                # calculate the offset from the center of the road
                y_eval = y_eval*ym_per_pix
                midpoint_car = self.im_shape[1]/2.0
                midpoint_lane =(right_fit_cr[0]*(y_eval**2) + right_fit_cr[1]*y_eval + right_fit_cr[2]) + \
                               (left_fit_cr[0]*(y_eval**2) + left_fit_cr[1]*y_eval + left_fit_cr[2])

                offset = midpoint_car*xm_per_pix - midpoint_lane/2

                # initialize the curvature and offset if this is the first detection
                if self.left_curverad == None:
                    self.left_curverad = left_curverad
                    self.right_curverad = right_curverad
                    self.offset = offset

                # average out the curvature and offset
                else:
                    self.left_curverad = self.left_curverad * 0.8 + left_curverad*0.2
                    self.right_curverad = self.right_curverad * 0.8 + right_curverad*0.2
                    self.offset = self.offset * 0.9 + offset*0.1

    # project results on the source image
    def project_on_road_debug(self, image_input):
        image = image_input[self.remove_pixels:, :]
        image = self.trans_per(image)
        self.im_shape = image.shape
        self.get_fit(image)

        if self.detected_first & self.detected:
            # create fill image
            temp_filler = np.zeros((self.remove_pixels,self.im_shape[1])).astype(np.uint8)
            filler = np.dstack((temp_filler,temp_filler,temp_filler))

            # create an image to draw the lines on
            warp_zero = np.zeros_like(image).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            ploty = np.linspace(0, image_input.shape[0]-1, image_input.shape[0] )
            left_fitx = self.best_fit_l[0]*ploty**2 + self.best_fit_l[1]*ploty + self.best_fit_l[2]
            right_fitx = self.best_fit_r[0]*ploty**2 + self.best_fit_r[1]*ploty + self.best_fit_r[2]

            # recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

            # warp the blank back to original image space using inverse perspective matrix (Minv)
            newwarp = cv2.warpPerspective(color_warp, self.Minv, color_warp.shape[-2:None:-1])
            left_right = cv2.warpPerspective(self.left_right, self.Minv, color_warp.shape[-2:None:-1])
            # combine the result with the original image
            left_right_fill = np.vstack((filler,left_right)) 
            result = cv2.addWeighted(left_right_fill,1, image_input, 1, 0)
            result = cv2.addWeighted(result, 1, np.vstack((filler,newwarp)), 0.3, 0)

            # get curvature and offset
            self.calculate_curvature_offset()

            # plot text on resulting image
            img_text = "radius of curvature: " + str(round((self.left_curverad + self.right_curverad)/2,2)) + ' (m)'

            if self.offset< 0:
                img_text2 = "vehicle is: " + str(round(np.abs(self.offset),2)) + ' (m) left of center'
            else:
                img_text2 = "vehicle is: " + str(round(np.abs(self.offset),2)) + ' (m) right of center'

            small = cv2.resize(left_right_fill, (0,0), fx=0.5, fy=0.5)
            small2 = cv2.resize(np.vstack((filler,self.left_right)), (0,0), fx=0.5, fy=0.5)

            result2 = cv2.resize(np.hstack((result, np.vstack((small2,small)))), (0,0), fx=self.enlarge, fy=self.enlarge)
            #result2 = cv2.resize(np.hstack((np.vstack((filler,np.dstack((self.binary_image*255,self.binary_image*255,self.binary_image*255)))), np.vstack((small2,small)))), (0,0), fx=self.enlarge, fy=self.enlarge)

            cv2.putText(result2,img_text, (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)
            cv2.putText(result2,img_text2,(15,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)

            return result2 

        # if lanes were not detected output source image
        else:
            return_image = cv2.resize(np.hstack((image_input,cv2.resize(np.zeros_like(image_input),(0,0), fx=0.5, fy=1.0))),(0,0), fx=self.enlarge, fy=self.enlarge)
            return return_image 

    # project results on the source image
    def project_on_road(self, image_input):
        image = image_input[self.remove_pixels:, :]
        image = self.trans_per(image)
        self.im_shape = image.shape
        self.get_fit(image)
        
        if self.detected_first & self.detected:
            # create fill image
            temp_filler = np.zeros((self.remove_pixels,self.im_shape[1])).astype(np.uint8)
            filler = np.dstack((temp_filler,temp_filler,temp_filler))

            # create an image to draw the lines on
            warp_zero = np.zeros_like(image).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            ploty = np.linspace(0, image_input.shape[0]-1, image_input.shape[0] )
            left_fitx = self.best_fit_l[0]*ploty**2 + self.best_fit_l[1]*ploty + self.best_fit_l[2]
            right_fitx = self.best_fit_r[0]*ploty**2 + self.best_fit_r[1]*ploty + self.best_fit_r[2]

            # recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

            # warp the blank back to original image space using inverse perspective matrix (Minv)
            newwarp = cv2.warpPerspective(color_warp, self.Minv, color_warp.shape[-2:None:-1])
            left_right = cv2.warpPerspective(self.left_right, self.Minv, color_warp.shape[-2:None:-1])
            # combine the result with the original image
            left_right_fill = np.vstack((filler,left_right)) 
            result = cv2.addWeighted(left_right_fill,1, image_input, 1, 0)
            result = cv2.addWeighted(result, 1, np.vstack((filler,newwarp)), 0.3, 0)


            # get curvature and offset
            self.calculate_curvature_offset()

            # plot text on resulting image
            img_text = "radius of curvature: " + str(round((self.left_curverad + self.right_curverad)/2,2)) + ' (m)'

            if self.offset< 0:
                img_text2 = "vehicle is: " + str(round(np.abs(self.offset),2)) + ' (m) left of center'
            else:
                img_text2 = "vehicle is: " + str(round(np.abs(self.offset),2)) + ' (m) right of center'

            result2 = cv2.resize(result, (0,0), fx=self.enlarge, fy=self.enlarge)

            cv2.putText(result2,img_text, (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)
            cv2.putText(result2,img_text2,(15,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)

            return result2

        # if lanes were not detected output source image
        else:
            return cv2.resize(image_input,(0,0), fx=self.enlarge, fy=self.enlarge)


lines = Lines()
lines.look_ahead = 10
lines.remove_pixels = 100
lines.enlarge = 2.25
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        
        # show the frame
        #lines.project_on_road_debug(image)
        cv2.imshow("Rpi lane detection", lines.project_on_road_debug(image))
        key = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame
        rawCapture.truncate()
        rawCapture.seek(0)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break

