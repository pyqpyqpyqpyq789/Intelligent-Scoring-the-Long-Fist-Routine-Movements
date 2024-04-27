import math
import numpy as np
import pandas as pd
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, median_absolute_error

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来显示中文标签
import shap


def Multi_ML(X_train, X_test, y_train, y_test):
    # 初始化回归模型
    linear_reg = LinearRegression()
    decision_tree_reg = DecisionTreeRegressor()
    random_forest_reg = RandomForestRegressor()
    svr_reg = SVR()
    knn = KNeighborsRegressor()
    lasso = Lasso()
    # mlp = MLPRegressor(hidden_layer_sizes=(34), early_stopping=True, alpha=1e-3, max_iter=50, learning_rate_init=0.1)
    bagging = BaggingRegressor()
    '''
    # 创建投票 VotingRegressor
    voting_regressor = VotingRegressor([('lasso', lasso),
                                        ('svr', svr_reg),
                                        ('knn', knn),
                                        ('tree', decision_tree_reg),
                                        ('rf', random_forest_reg),
                                        ('bag', bagging)],
                                       weights=[0.05, 0.05, 0.05, 0.15, 0.35, 0.35])
    '''

    # 训练模型
    linear_reg.fit(X_train, y_train)
    decision_tree_reg.fit(X_train, y_train)
    random_forest_reg.fit(X_train, y_train)
    svr_reg.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    lasso.fit(X_train, y_train)
    # mlp.fit(X_train, y_train)
    bagging.fit(X_train, y_train)
    # voting_regressor.fit(X_train, y_train)

    # 模型预测
    linear_pred = linear_reg.predict(X_test)
    decision_tree_pred = decision_tree_reg.predict(X_test)
    random_forest_pred = random_forest_reg.predict(X_test)
    svr_pred = svr_reg.predict(X_test)
    knn_pred = knn.predict(X_test)
    lasso_pred = lasso.predict(X_test)
    # mlp_pred = mlp.predict(X_test)
    bagging_pred = bagging.predict(X_test)

    # voting_pred = voting_regressor.predict(X_test)

    # 特征重要性

    # 计算Permutation Importance
    '''
    svr_importance = permutation_importance(svr_reg, X_test, y_test, n_repeats=30, random_state=42)
    # linear_importance = linear_reg.coef_
    linear_importance = permutation_importance(linear_reg, X_test, y_test, n_repeats=30, random_state=42)
    decision_tree_importance = permutation_importance(decision_tree_reg, X_test, y_test, n_repeats=30, random_state=42)
    random_forest_importance = permutation_importance(random_forest_reg, X_test, y_test, n_repeats=30, random_state=42)
    knn_importance = permutation_importance(knn, X_test, y_test, n_repeats=30, random_state=42)
    lasso_importance = permutation_importance(lasso, X_test, y_test, n_repeats=30, random_state=42)
    # mlp_importance = permutation_importance(mlp, X_test, y_test, n_repeats=30, random_state=42)
    bagging_importance = permutation_importance(bagging, X_test, y_test, n_repeats=30, random_state=42)
    '''
    # vot_feature_importance = permutation_importance(voting_regressor, X_test, y_test, n_repeats=30, random_state=42).importances_mean
    # vot_feature_importance = permutation_importance(voting_regressor, X_train, y_train, n_repeats=30, random_state=42).importances_mean

    '''
    plot_permutation_importance(linear_importance, 'Linear Regression')
    plot_permutation_importance(decision_tree_importance, 'Decision Tree Regression')
    plot_permutation_importance(random_forest_importance, 'Random Forest Regression')
    plot_permutation_importance(svr_importance, 'SVR')
    plot_permutation_importance(knn_importance, 'KNN')
    plot_permutation_importance(lasso_importance, 'LASSO')
    plot_permutation_importance(mlp_importance, 'MLP')
    plot_permutation_importance(bagging_importance, 'Bagging')
    '''

    # plot_vot_importance(imp=vot_feature_importance)

    # plot_shap(X_train=X_train, X_test=X_test, model=knn)
    # explainer = shap.KernelExplainer(knn.predict, X_train)
    # # 计算SHAP值
    # shap_values = explainer.shap_values(X_test)
    # print("shap_values.shape", shap_values.shape)
    # # 1. 绘制摘要图
    # shap.summary_plot(shap_values, X_test, feature_names=[f'Feature {ii}' for ii in range(X_test.shape[1])])

    return linear_pred, decision_tree_pred, random_forest_pred, svr_pred, knn_pred, lasso_pred, bagging_pred,  # voting_pred


# 绘制特征重要性
def plot_permutation_importance(importance, model_name):
    plt.figure(figsize=(10, 6))
    sorted_idx = importance.importances_mean.argsort()
    # plt.barh(range(n_features), importance.importances_mean[sorted_idx], tick_label=[f'Feature {i}' for i in sorted_idx])
    plt.barh(range(n_features), importance.importances_mean[sorted_idx],
             tick_label=[f'{feature_names[i]}' for i in sorted_idx])
    plt.title(f'Permutation Importance - {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.savefig(f'{model_name}.jpg')
    plt.show()


def plot_vot_importance(imp):
    print(imp.shape)
    imp = np.reshape(imp, (X.shape[1], X.shape[2]))

    act_names = [
        '预备势', '并步抱拳礼', '左右侧冲拳', '开步推掌翻掌抱拳', '震脚砸拳1', '蹬脚冲拳1',
        '马步左右冲拳', '震脚砸拳2', '蹬脚冲拳2', '马步右左冲拳', '插步摆掌', '勾手推掌',
        '弹踢推掌', '弓步冲拳1', '抡臂砸拳', '弓步冲拳2', '震脚弓步双推掌', '抡臂拍脚',
        '弓步顶肘', '歇步冲拳', '提膝穿掌1', '仆步穿掌1', '虚步挑掌', '震脚提膝上冲拳',
        '弓步架拳1', '蹬腿架拳', '转身提膝双挑掌', '提膝穿掌2', '仆步穿掌2', '仆步抡拍',
        '弓步架拳2', '收势']
    feat_names = [
        '左肘关节角', '右肘关节角', '左肩关节角', '右肩关节角', '左膝关节角', '右膝关节角', '左髋关节角', '右髋关节角',
        '肩髋扭转角', '肩髋扭转角',
        '左上臂倾角', '右上臂倾角', '左前臂倾角', '右前臂倾角', '左大腿倾角', '右大腿倾角', '左小腿倾角', '右小腿倾角',
    ]
    import seaborn as sns
    plt.figure(figsize=(10, 8))  # 可选的，设置图形大小
    sns.heatmap(abs(imp), annot=False, cmap='Blues')  # annot=True用于显示数值，cmap指定颜色地图
    plt.title("特征重要性")
    # plt.xlabel("X-Axis")
    # plt.ylabel("Y-Axis")
    plt.xlabel("指标")
    plt.ylabel("动作")
    plt.xticks([pos + 0.5 for pos in range(len(feat_names))], feat_names)
    plt.yticks([pos + 0.5 for pos in range(len(act_names))], act_names)
    # 旋转x轴标签文本
    for label in plt.gca().get_xticklabels():
        label.set_rotation(90)  # 设置旋转角度为30度

    # 旋转y轴标签文本
    for label in plt.gca().get_yticklabels():
        label.set_rotation(0)  # 设置旋转角度为-30度，使标签向y方向旋转
    plt.show()


def plot_shap(X_train, X_test, model):
    # shap值
    # 初始化shap解释器
    # explainer = shap.KernelExplainer(model.predict_proba, X_train)
    explainer = shap.Explainer(model.predict, X_train)
    # explainer = shap.KernelExplainer(model, X_train)

    # 计算测试集上的shap值
    # shap_values = explainer.shap_values(X_test)
    shap_values = explainer.shap_values(X_train)

    # 1. 绘制摘要图
    shap.summary_plot(shap_values, X_train, feature_names=[f'Feature {ii}' for ii in range(X_test.shape[1])])
    # 2. 绘制强调图（选择第一个样本进行演示）
    sample_index = 0
    # shap.save_html('force_plot.html',
    #                shap.force_plot(explainer.expected_value, shap_values[sample_index], X_test[sample_index],
    #                                feature_names=[f'Feature {ii}' for ii in range(X_test.shape[1])]))
    shap.save_html('force_plot.html',
                   shap.force_plot(base_value=model.predict(X_test[sample_index]),
                                   shap_values=shap_values[sample_index],
                                   features=X_test[sample_index],
                                   feature_names=[f'Feature {ii}' for ii in range(X_test.shape[1])]))
    # 3. 绘制依赖图（选择第一个特征进行演示）
    if isinstance(shap_values, list):
        # 处理 shap_values 是列表的情况
        for class_idx in range(len(shap_values)):
            shap.dependence_plot(0, shap_values[class_idx], X_test,
                                 feature_names=[f'Feature {ii}' for ii in range(X_test.shape[1])])
    else:
        # 当 shap_values 是数组时
        shap.dependence_plot(0, shap_values, X_test, feature_names=[f'Feature {ii}' for ii in range(X_test.shape[1])])


def get_weight(RMSE_arr):
    weight = []
    # RMSE_arr = (RMSE_arr-20) * 5
    RMSE_arr = (math.e ** abs(RMSE_arr - max(RMSE_arr))) ** 30
    # RMSE_arr = np.tan(abs(RMSE_arr - max(RMSE_arr)))
    print('RMSE_arr', RMSE_arr)
    for a in range(len(RMSE_arr)):
        # w_a = (1/RMSE_arr[a]) / sum((1/RMSE_arr))
        # w_a = (math.e ** (-RMSE_arr[a])) / sum((math.e ** (-1 * RMSE_arr)))
        # w_a = (1/(math.e ** RMSE_arr[a])) / sum(1/(math.e ** RMSE_arr))
        # w_a = (math.e ** (-RMSE_arr[a])) / sum((math.e ** (-1 * RMSE_arr)))
        w_a = RMSE_arr[a] / sum(RMSE_arr)
        weight.append(w_a)
    # print("sum(weight)=", sum(weight))
    return weight


def VOTING_Model(X_train, X_test, y_train, y_test, RMSE_arr):
    weight = get_weight(RMSE_arr)
    # 初始化回归模型
    linear_reg = LinearRegression()
    decision_tree_reg = DecisionTreeRegressor()
    random_forest_reg = RandomForestRegressor()
    svr_reg = SVR()
    knn = KNeighborsRegressor()
    lasso = Lasso()
    # mlp = MLPRegressor(hidden_layer_sizes=(34), early_stopping=True, alpha=1e-3, max_iter=50, learning_rate_init=0.1)
    bagging = BaggingRegressor()
    # 创建投票 VotingRegressor
    voting_regressor = VotingRegressor([('linear', linear_reg),
                                        ('tree', decision_tree_reg),
                                        ('rf', random_forest_reg),
                                        ('svr', svr_reg),
                                        ('knn', knn),
                                        ('lasso', lasso),
                                        ('bag', bagging)],
                                       weights=weight)
    # 训练模型
    voting_regressor.fit(X_train, y_train)
    voting_pred = voting_regressor.predict(X_test)


    # vot_feature_importance = permutation_importance(voting_regressor, X_train, y_train, n_repeats=30,
    #                                                 random_state=42).importances_mean
    # plot_vot_importance(imp=vot_feature_importance)
    # plot_shap(X_train=X_train, X_test=X_test, model=voting_regressor) ####shap值
    return voting_pred, weight


def sMAPE(y_true_, y_pred_):   # MAPE计算公式
    # return np.mean(np.abs((y_pred_ - y_true_) / y_true_))*100
    return np.mean(2 * np.abs(y_pred_ - y_true_) / (np.abs(y_true_) + np.abs(y_pred_)))*100


def Evaluate(pred_list, model_names):
    eva_results = []
    # dul_list = zip()
    for prediction, names in zip(pred_list, model_names):
        eva_results.append([names,
                            '{:.3f}'.format(median_absolute_error(y, prediction)),
                            '{:.3f}'.format(np.sqrt(mean_squared_error(y, prediction))),
                            '{:.3f}'.format(sMAPE(y, prediction)),
                            '{:.3f}'.format(r2_score(y, prediction)),
                            pearsonr(y, prediction)[0],
                            pearsonr(y, prediction)[1]])
    eva_results = pd.DataFrame(eva_results)
    eva_results.columns = ['算法', 'MAE', 'RMSE', 'sMAPE', 'R²', 'correlation', 'p_value']
    # 根据P值标注显著性
    for e in range(len(eva_results['p_value'])):
        if 0.01 <= float(eva_results['p_value'][e]) < 0.05:
            eva_results['p_value'][e] = str(eva_results['p_value'][e])+'*'
        elif 0.001 <= float(eva_results['p_value'][e]) < 0.01:
            eva_results['p_value'][e] = str(eva_results['p_value'][e])+'**'
        elif float(eva_results['p_value'][e]) < 0.001:
            eva_results['p_value'][e] = str(eva_results['p_value'][e])+'***'
    print(eva_results)
    eva_results[['算法', 'MAE', 'RMSE', 'sMAPE', 'R²', 'correlation', 'p_value']].to_csv("ML_evaluate_results.csv", index=False)
    return eva_results


def perd_scatter_plot(pred_list_, model_name_list_, RR):
    for index in range(len(pred_list_)):
        # 创建散点图
        plt.figure(figsize=(8, 6))
        # 绘制45度对角线
        plt.plot([min(y), max(y)], [min(y), max(y)], linestyle='--', color='red',
                 linewidth=2)
        plt.scatter(y, pred_list_[index], marker='x', c='blue')#, label='实验样本')
        # 添加标签和标题
        plt.xlabel('true value', fontsize=10)
        plt.ylabel('predict value', fontsize=10)
        plt.text(min(y), max(y), 'R^2='+str(RR[index]), fontsize=10)
        plt.title(model_name_list_[index])
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # 读取数据

    y = np.load('QualityScore.npy')  # shape (176,)
    X = np.load('XingShenQuan.npy')[:, :, :18]  # shape (176, 60, 11)
    # X = np.load('DengJianGeCaiYang.npy')[:, :, :18]

    n_features = X.shape[-1]
    # for i in range(len(y)):
    #     for j in range(n_features):
    #         X[i, :, j] = TimeSeriesScalerMeanVariance().fit_transform(X[i, :, j])  #标准化

    # 将特征展平
    flatten_X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))  # (样本量n, 时间/5 ✖ 特征通道)

    # 数据归一化
    # scaler = StandardScaler()
    # X_sub1_scaled = scaler.fit_transform(X_sub1)
    # X_sub2_scaled = scaler.transform(X_sub2)

    # 留一法测试精度
    linearPred, decision_treePred, random_forestPred, svrPred, knnPred, lassoPred, baggingPred, votingPred = [], [], [], [], [], [], [], []
    for i in range(y.shape[0]):
        Test_y = np.array([y[i]])
        print('Test_y', Test_y.shape)
        Test_x = np.array([flatten_X[i]])
        # Test_x = flatten_X[i]
        print("Test_x.shape", Test_x.shape)
        Train_y = np.delete(y, i, axis=0)
        Train_x = np.delete(flatten_X, i, axis=0)
        print('train_x', Train_x.shape)
        linear_pred, decision_tree_pred, random_forest_pred, svr_pred, knn_pred, lasso_pred, bagging_pred = Multi_ML(
            Train_x, Test_x, Train_y, Test_y)
        linearPred.append(linear_pred)
        decision_treePred.append(decision_tree_pred)
        random_forestPred.append(random_forest_pred)
        svrPred.append(svr_pred)
        knnPred.append(knn_pred)
        lassoPred.append(lasso_pred)
        baggingPred.append(bagging_pred)
        # votingPred.append(voting_pred)

    # 计算均方根误差（RMSE）
    rmse1 = np.sqrt(mean_squared_error(y, linearPred))
    rmse2 = np.sqrt(mean_squared_error(y, decision_treePred))
    rmse3 = np.sqrt(mean_squared_error(y, random_forestPred))
    rmse4 = np.sqrt(mean_squared_error(y, svrPred))
    rmse5 = np.sqrt(mean_squared_error(y, knnPred))
    rmse6 = np.sqrt(mean_squared_error(y, lassoPred))
    # rmse7 = np.sqrt(mean_squared_error(y_test, mlp_pred))
    rmse8 = np.sqrt(mean_squared_error(y, baggingPred))
    # rmse9 = np.sqrt(mean_squared_error(y, votingPred))
    RMSE_arr = np.array([rmse1, rmse2, rmse3, rmse4, rmse5, rmse6, rmse8])
    Weights = 0
    for i in range(y.shape[0]):
        Test_y, Test_x, Train_y, Train_x = np.array([y[i]]), np.array([flatten_X[i]]), np.delete(y, i, axis=0), np.delete(flatten_X, i, axis=0)

        voting_pred, Weights = VOTING_Model(Train_x, Test_x, Train_y, Test_y, RMSE_arr)
        votingPred.append(voting_pred)
        print("Weights", Weights)
    rmse9 = np.sqrt(mean_squared_error(y, votingPred))

    pred_list = [linearPred, decision_treePred, random_forestPred, svrPred, knnPred, lassoPred, baggingPred, votingPred]
    model_name_list = ['Linear', 'DT', 'RF', 'SVR', 'K-NN', 'Lasso', 'Bagging', 'Voting']
    eva_results = Evaluate(pred_list=pred_list, model_names=model_name_list)
    perd_scatter_plot(pred_list_=pred_list, model_name_list_=model_name_list, RR=eva_results['R²'])
    # # 创建散点图
    # plt.figure(figsize=(8, 6))
    # # 绘制45度对角线
    # plt.plot([min(y), max(y)], [min(y), max(y)], linestyle='--', color='red',
    #          linewidth=2)
    # plt.scatter(y, votingPred, marker='x', c='blue', label='实验样本')
    # # 添加标签和标题
    # plt.xlabel('人工评分')
    # plt.ylabel('模型评分')
    # plt.title('多模型决策的回归结果')
    # plt.legend()
    # plt.show()

    # 绘制RMSE—权重图
    plt.plot(sorted(RMSE_arr, key=float), sorted(Weights, key=float, reverse=True))
    plt.scatter(RMSE_arr, Weights)
    # 添加标签和标题
    plt.xlabel('RMSE', fontsize=10)
    plt.ylabel('Weight', fontsize=10)
    plt.show()
