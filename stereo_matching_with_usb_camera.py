# -- coding: utf-8 -- 
# @Author:  Kaixiang Song
# @Project: VideoCapture 
# @File:    stereo_camera.py
# @Time:    2020/2/7 17:43

import cv2
import numpy as np


def get_camera(width=640, height=480, usb_camera=True):
    """
    Get My Camera.
    :param height: Set height.
    :param width: Set Width.
    :param usb_camera: Using USB camera, True or False.
    :return:
    """
    width *= 2                                      # 双目摄像头，总宽度
    cap = cv2.VideoCapture(int(usb_camera))         # VideoCapture()中参数是1，表示打开外接usb摄像头
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)        # 设置摄像头的分辨率，宽
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)      # 设置摄像头的分辨率，高
    return cap


def get_camera_info(cap):
    """
    Get Camera Information.
    :param cap: Video Capture.
    :return:
    """
    data_dict = {'宽': cv2.CAP_PROP_FRAME_WIDTH,
                 '高': cv2.CAP_PROP_FRAME_HEIGHT,
                 '帧率': cv2.CAP_PROP_FPS,
                 }

    print('\n摄像头信息：')
    for k, v in data_dict.items():
        print("{}: {}.".format(k, cap.get(v)))


def real_time_video(cap):
    """
    Only show stereo images.
    :param cap: Video Capture.
    :return:
    """
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(frame.shape)
            width = frame.shape[1] // 2
            cv2.imshow('left', frame[:, :width, :])
            cv2.imshow('right', frame[:, width:, :])
            cv2.waitKey(1)


class SGM:
    """
    Stereo Matching With SGM.
    """
    def __init__(self):

        def nothing(x):
            pass

        # Parameters.
        window_size = 5
        self.min_disp = 16
        self.num_disp = 192 - self.min_disp
        uniquenessRatio = 1
        speckleRange = 3
        speckleWindowSize = 3
        disp12MaxDiff = 200
        P1 = 600
        P2 = 2400

        # Adjust parameters in window.
        cv2.namedWindow('disparity')
        cv2.createTrackbar('speckleRange', 'disparity', speckleRange, 50, nothing)
        cv2.createTrackbar('window_size', 'disparity', window_size, 21, nothing)
        cv2.createTrackbar('speckleWindowSize', 'disparity', speckleWindowSize, 200, nothing)
        cv2.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50, nothing)
        cv2.createTrackbar('disp12MaxDiff', 'disparity', disp12MaxDiff, 250, nothing)

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disp,
            blockSize=window_size,
            uniquenessRatio=uniquenessRatio,
            speckleRange=speckleRange,
            speckleWindowSize=speckleWindowSize,
            disp12MaxDiff=disp12MaxDiff,
            P1=P1,
            P2=P2
        )

    def update_and_compute(self, left, right):
        self.stereo.setBlockSize(cv2.getTrackbarPos('window_size', 'disparity'))
        self.stereo.setUniquenessRatio(cv2.getTrackbarPos('uniquenessRatio', 'disparity'))
        self.stereo.setSpeckleWindowSize(cv2.getTrackbarPos('speckleWindowSize', 'disparity'))
        self.stereo.setSpeckleRange(cv2.getTrackbarPos('speckleRange', 'disparity'))
        self.stereo.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff', 'disparity'))

        print('computing disparity...')
        disp = self.stereo.compute(left, right).astype(np.float32) / 16.0

        cv2.imshow('disparity', (disp - self.min_disp) / self.num_disp)


def real_time_disparity(cap):
    sgm = SGM()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w, _ = frame.shape
            left, right = frame[:, :(w // 2), :], frame[:, (w // 2):, :]
            cv2.imshow('left', left)
            cv2.imshow('right', right)
            sgm.update_and_compute(left, right)
            cv2.waitKey(1)


if __name__ == "__main__":
    """
    一些设定的分辨率格式：1280*480，640*240等
    """
    cap = get_camera(1280, 720)
    get_camera_info(cap)
    # real_time_video(cap)
    real_time_disparity(cap)



