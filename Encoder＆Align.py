import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# from typing import List, Mapping, Optional, Tuple, Union

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import math
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def get_angle(p1, p2, p3, p4):
    vector1 = p2 - p1
    vector2 = p3 - p4
    cos = np.dot(vector1, vector2) / np.sqrt(sum(vector1 ** 2 + vector2 ** 2))
    return np.arccos(cos) * 180 / np.pi


def get_vertical_angle(p1, p2):
    vector1 = p1 - p2
    cos = np.dot(vector1, np.array([0, 1, 0])) / np.sqrt(sum(vector1 ** 2 + np.array([0, 1, 0]) ** 2))
    return np.arccos(cos) * 180 / np.pi


def Extract_feature(a):
    print('a', a.shape)
    a = a.astype(float)
    # 构造邻接表，表示要计算哪些角度
    bone_vector_list = [[6, 5, 5, 4],  # 左肘
                        [10, 9, 9, 8],  # 右肘
                        [5, 4, 4, 20],  # 左肩
                        [9, 8, 8, 20],  # 右肩
                        [12, 13, 13, 14],  # 左膝
                        [16, 17, 17, 18],  # 右膝
                        [0, 12, 12, 13],  # 左髋
                        [0, 16, 16, 17],  # 右髋
                        [4, 8, 12, 16],  # 肩髋扭转
                        [2, 3, 12, 16]]  # 头髋扭转
    # 各部位与垂直轴夹角
    vertical_angle_list = [[4, 5],  # 左臂
                           [8, 9],  # 右臂
                           [5, 6],  # 左前臂
                           [9, 10],  # 右前臂
                           [12, 13],  # 左腿
                           [16, 17],  # 右腿
                           [13, 14],  # 左小腿
                           [17, 18]  # 右小腿
                           ]
    # Spatial_Feature = np.zeros((a.shape[1], len(bone_vector_list) + len(vertical_angle_list)))
    Spatial_Feature = np.zeros((a.shape[1], len(bone_vector_list) + len(vertical_angle_list) + len(bone_vector_list)))
    print("Spatial_Feature.shape:", Spatial_Feature.shape)
    for i in range(a.shape[1]):
        for j in range(len(bone_vector_list)):
            Spatial_Feature[i, j] = get_angle(p1=a[:, i, bone_vector_list[j][0]],
                                              p2=a[:, i, bone_vector_list[j][1]],
                                              p3=a[:, i, bone_vector_list[j][2]],
                                              p4=a[:, i, bone_vector_list[j][3]])
    for i in range(a.shape[1]):
        for j in range(len(vertical_angle_list)):
            Spatial_Feature[i, len(bone_vector_list) + j] = get_vertical_angle(p1=a[:, i, vertical_angle_list[j][0]],
                                                                               p2=a[:, i, vertical_angle_list[j][1]])
    for i in range(a.shape[1]):
        for j in range(len(bone_vector_list)):
            if j != 0:
                Spatial_Feature[i, len(bone_vector_list) + len(vertical_angle_list) + j] = (Spatial_Feature[i, j] - Spatial_Feature[i-1, j]) * 60
    df = pd.DataFrame(Spatial_Feature)
    Spatial_Feature = df.interpolate(method='linear')
    Spatial_Feature = Spatial_Feature.fillna(0).to_numpy()
    return Spatial_Feature


def add_weight(angle_arr):  # 为重要的角度加权
    weighted_angle_arr = angle_arr
    important_angle = [4, 6, 13, 16]
    for k in important_angle:
        weighted_angle_arr[:, k] = angle_arr[:, k] * 1.0
    return weighted_angle_arr


def draw_path(test_angle, data_, path):
    pca = PCA(n_components=1, copy=True)
    print('data', data_.shape, 'test_angle', test_angle.shape)
    new_z1 = pca.fit_transform(data_)
    new_test_angle1 = pca.transform(test_angle)

    new_z1 = new_z1 + 60  # 拉开距离方便可视化
    plt.figure(figsize=(20, 8))
    # for i in range(len(path)):
    #     # 用降维的一维序列做可视化
    #     plt.plot([path[i][0], path[i][1]], [new_test_angle1[path[i][0]], new_z1[path[i][1]]], linewidth=0.1, c='r')

    f = 0
    for i in range(len(path)):
        if path[i][1] in list(Keyframes.values()):
            # 用降维的一维序列做可视化
            if f == 0:
                plt.plot([path[i][0], path[i][1]], [new_test_angle1[path[i][0]], new_z1[path[i][1]]], linewidth=1,
                         linestyle="--", c='k', label='匹配线')
            else:
                plt.plot([path[i][0], path[i][1]], [new_test_angle1[path[i][0]], new_z1[path[i][1]]], linewidth=1,
                         linestyle="--", c='k')
            f += 1

    plt.plot(np.arange(0, len(new_test_angle1), 1), new_test_angle1, c="b", label='测试动作序列')
    plt.plot(np.arange(0, len(new_z1), 1), new_z1, c="y", label='模板动作序列')
    plt.legend()
    plt.show()


def Alignment(test_angle, data_):
    # distance, path = fastdtw(new_test_angle, new_z, dist=euclidean)  # dtw

    distance, path = fastdtw(add_weight(test_angle), add_weight(data_), dist=euclidean)  # dtw
    print('path:', path)
    draw_path(test_angle=test_angle, data_=data_, path=path)

    my_key_frames = []
    # suggest = []
    for key, value in Keyframes.items():
        print('key',  key, 'value', value)
        index = np.where(np.array(path)[:, 1] == value)
        for i in index:
            for k in range(15):  # 逐一比较前后20帧与标准动作的余弦相似度，对关键帧重定位
                jx = np.array(path)[i, 0].tolist()[0] - 10 + k
                similarity1 = abs(pdist([data_[value], test_angle[jx]], "cosine"))
                similarity2 = abs(pdist([data_[value], test_angle[jx + 1]], "cosine"))
                # similarity1 = Compare(action1=test_angle[jx], action2=data[value], name_=key)
                # similarity2 = Compare(action1=test_angle[jx + 1], action2=data[value], name_=key)
                if similarity1 < similarity2:
                    kf = jx
                else:
                    kf = jx + 1
            my_key_frames.append([str(key), jx])
            jj = kf
            break
        # suggest.append([str(key), data_[value] - test_angle[jj]])
    # print(my_key_frames)
    print(len(my_key_frames))
    # print(suggest)

    return my_key_frames


if __name__ == '__main__':
    #   读入标准动作
    input_file = 'new_KPTs/'
    out_file = 'different_sampling/'
    # data = Extract_feature(a=np.load(input_file + 'WeChat_20240108113352.npy', allow_pickle=True))
    data = Extract_feature(a=np.load(input_file + 'WeChat_20240108113419.npy', allow_pickle=True))
    print(data.shape)
    z = data
    point_num = 32  # [16, 24, 32, 40, 48, 56]
    length = z.shape[0]
    indices = np.linspace(0, length - 1, point_num, endpoint=False, dtype=int)

    Keyframe_ = []
    for i in range(point_num):
        Keyframe_.append((str(i), int(indices[i])))
    Keyframes = {}
    for key, value in Keyframe_:
        Keyframes[key] = value

    Keyframes = {
        '预备势': int(indices[0]),
        '并步抱拳礼': int(indices[1]),
        '左右侧冲拳': int(indices[2]),
        '开步推掌翻掌抱拳': int(indices[3]),
        '震脚砸拳1': int(indices[4]),
        '蹬脚冲拳1': int(indices[5]),
        '马步左右冲拳': int(indices[6]),
        '震脚砸拳2': int(indices[7]),
        '蹬脚冲拳2': int(indices[8]),
        '马步右左冲拳': int(indices[9]),
        '插步摆掌': int(indices[10]),
        '勾手推掌': int(indices[11]),
        '弹踢推掌': int(indices[12]),
        '弓步冲拳1': int(indices[13]),
        '抡臂砸拳': int(indices[14]),
        '弓步冲拳2': int(indices[15]),
        '震脚弓步双推掌': int(indices[16]),
        '抡臂拍脚': int(indices[17]),
        '弓步顶肘': int(indices[18]),
        '歇步冲拳': int(indices[19]),
        '提膝穿掌1': int(indices[20]),
        '仆步穿掌1': int(indices[21]),
        '虚步挑掌': int(indices[22]),
        '震脚提膝上冲拳': int(indices[23]),
        '弓步架拳1': int(indices[24]),
        '蹬腿架拳': int(indices[25]),
        '转身提膝双挑掌': int(indices[26]),
        '提膝穿掌2': int(indices[27]),
        '仆步穿掌2': int(indices[28]),
        '仆步抡拍': int(indices[29]),
        '弓步架拳2': int(indices[30]),
        '收势': int(indices[31])}  # 从模板中指定动作最标准的关键帧

    #   读入测试数据
    # (3, length, 33)
    # input_file = 'output/'
    # (1815, 3, 300, 25, 2)

    dataset_for_exam = np.zeros((len(os.listdir(input_file)), point_num, 28))
    dataset2_for_exam = np.zeros((len(os.listdir(input_file)), point_num, 28))
    align_kpts = np.zeros((len(os.listdir(input_file)), 3, point_num, 33))
    QualityScore = np.zeros(len(os.listdir(input_file)))

    # Scores = np.load('WUSHI_score.npy')  #[9.43, 9.57, 8.67, 8.07, 7.23, 8.5, 6.77, 7.43, 6.85, 6.1, 4.97, 5.85]

    flag = 0
    for file in os.listdir(input_file):
        print(file)
        # QualityScore[flag] = Scores[int(file[1:3])-1]

        full_path = input_file + file   #.split('.')[0] + '.npy'
        TEST = np.load(full_path, allow_pickle=True)#[:, :, :, 0]
        print('TEST.shape', TEST.shape)
        Test_Angle = Extract_feature(a=TEST)

        My_Key_Frames = Alignment(test_angle=Test_Angle[:, :18], data_=data[:, :18])
        for k in range(len(My_Key_Frames)):
            dataset_for_exam[flag, k] = Test_Angle[My_Key_Frames[k][-1]]
            align_kpts[flag, :, k] = TEST[:, My_Key_Frames[k][-1]]
        print('np.sum(np.isnan(dataset_for_exam))', np.sum(np.isnan(dataset_for_exam)))
        # 使用np.linspace生成point_num个等分点
        length = Test_Angle.shape[0]
        indices = np.linspace(0, length - 1, point_num, endpoint=False, dtype=int)
        print("indices", indices)
        dataset2_for_exam[flag] = Test_Angle[indices]
        # print("aa.shape", aa.shape)
        flag += 1
    print('dataset_for_exam.shape', dataset_for_exam.shape)
    # np.save(out_file+'Aligned_XSQ_kpts_'+str(point_num)+'.npy', dataset_for_exam)
    
