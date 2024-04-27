import cv2
import mediapipe as mp
import os
import numpy as np


# mediapipe 模型变量初始化
def mediapipe_varibles_init():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2, smooth_landmarks=True)
    mp_drawing = mp.solutions.drawing_utils
    return pose, mp_pose, mp_drawing


# 在3D里画出骨架的函数
def draw_3d_pose():
    X, Y, Z = [], [], []
    results = pose.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    if results.pose_world_landmarks:

        for i in range(33):
            pos_x = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark(i).value].x  # * -cap.get(3)
            pos_y = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark(i).value].y  # * -cap.get(4)
            pos_z = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark(i).value].z  # * -cap.get(3)

            X.append(pos_x)
            Y.append(pos_y)
            Z.append(pos_z)

    return np.array(X), np.array(Y), np.array(Z)


if __name__ == "__main__":
    action_name = '左右侧冲拳/'
    input_file = "裁剪视频/"+action_name
    output_file = "关节轨迹/"+action_name
    if not os.path.exists(output_file):
        os.mkdir(output_file)

    for file in os.listdir(input_file):
        print(file)
        X_, Y_, Z_ = [], [], []
        pose, mp_pose, mp_drawing = mediapipe_varibles_init()
        # 打开摄像头，0是第一个摄像头，如果想换一个摄像头请改变这个数字
        cap = cv2.VideoCapture(input_file+file)     #读取视频
        fps = cap.get(cv2.CAP_PROP_FPS)     #获得帧率

        while True:
            # 获取每一帧的图像
            _, f = cap.read()
            # 如果没有提取到图像，结束整个循环
            if f is None:
                break
            X, Y, Z = draw_3d_pose()
            X_.append(X)
            Y_.append(Y)
            Z_.append(Z)
            print('X_[-1]', X_[-1])
            print('len X_[-1]', len(X_[-1]))

        # 调用窗口关闭函数
        cap.release()
        X_, Y_, Z_ = np.array(X_), np.array(Y_), np.array(Z_)
        print(np.array([X_, Y_, Z_]).shape)
        np.save(output_file+file.split('.')[0]+'.npy', np.array([X_, Y_, Z_]))


"""
A NamedTuple with fields describing the landmarks on the most prominate
      person detected:
        1) "pose_landmarks" field that contains the pose landmarks.
        2) "pose_world_landmarks" field that contains the pose landmarks in
        real-world 3D coordinates that are in meters with the origin at the
        center between hips.
        3) "segmentation_mask" field that contains the segmentation mask if
           "enable_segmentation" is set to true.
"""