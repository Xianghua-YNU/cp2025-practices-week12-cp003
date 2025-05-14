import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def load_bacterial_data(file_path):
    """
    从文件中加载细菌生长实验数据
    
    参数:
        file_path (str): 数据文件路径，要求是CSV格式，包含两列数据
        
    返回:
        tuple: 包含两个numpy数组的元组 (时间数据, 酶活性测量值)
    """
    # 使用numpy读取文本文件数据，假设是逗号分隔
    data = np.loadtxt(file_path, delimiter=',')
    t = data[:, 0]  # 提取第一列作为时间数据
    activity = data[:, 1]  # 提取第二列作为酶活性数据
    return t, activity

def V_model(t, tau):
    """
    V(t)模型函数 - 描述细菌生长初期阶段的动力学模型
    
    参数:
        t (float or numpy.ndarray): 时间点或时间序列
        tau (float): 时间常数，控制曲线上升速率
        
    返回:
        float or numpy.ndarray: 在给定时间点的模型预测值
        模型公式: V(t) = 1 - exp(-t/τ)
    """
    return 1 - np.exp(-t / tau)

def W_model(t, A, tau):
    """
    W(t)模型函数 - 描述细菌生长后期阶段的动力学模型
    
    参数:
        t (float or numpy.ndarray): 时间点或时间序列
        A (float): 比例系数，控制曲线幅度
        tau (float): 时间常数，与V模型中的τ相同
        
    返回:
        float or numpy.ndarray: 在给定时间点的模型预测值
        模型公式: W(t) = A * [exp(-t/τ) - 1 + t/τ]
    """
    return A * (np.exp(-t / tau) - 1 + t / tau)

def fit_model(t, data, model_func, p0):
    """
    使用非线性最小二乘法拟合模型参数
    
    参数:
        t (numpy.ndarray): 实验时间数据
        data (numpy.ndarray): 实验测量数据
        model_func (function): 要拟合的模型函数
        p0 (list): 模型参数的初始猜测值
        
    返回:
        tuple: (最优参数数组popt, 参数协方差矩阵pcov)
    """
    # 使用scipy的curve_fit进行参数拟合
    popt, pcov = curve_fit(model_func, t, data, p0=p0)
    return popt, pcov

def plot_results(t, data, model_func, popt, title):
    """
    可视化展示实验数据与模型拟合结果
    
    参数:
        t (numpy.ndarray): 实验时间数据
        data (numpy.ndarray): 实验测量数据
        model_func (function): 拟合使用的模型函数
        popt (numpy.ndarray): 拟合得到的最优参数
        title (str): 图表标题
    """
    plt.figure(figsize=(10, 6))  # 创建图形窗口
    plt.plot(t, data, 'o', label='Experimental data')  # 绘制实验数据点
    t_fit = np.linspace(min(t), max(t), 1000)  # 生成密集的时间点用于绘制平滑曲线
    plt.plot(t_fit, model_func(t_fit, *popt), '-', label='Model fit')  # 绘制拟合曲线
    plt.xlabel('Time')  # x轴标签
    plt.ylabel('Activity')  # y轴标签
    plt.title(title)  # 图表标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()  # 展示图形

if __name__ == "__main__":
    # 主程序执行部分
    
    # 数据加载部分
    # 注意：实际使用时需要确保这些路径指向真实的数据文件
    data_dir = "/Users/lixh/Library/CloudStorage/OneDrive-个人/Code/cp2025-InterpolateFit/细菌生长实验数据拟合"
    t_V, V_data = load_bacterial_data(f"{data_dir}/g149novickA.txt")  # 加载V模型数据
    t_W, W_data = load_bacterial_data(f"{data_dir}/g149novickB.txt")  # 加载W模型数据
    
    # V(t)模型拟合
    # 初始参数猜测: [τ=1.0]
   # V(t)模型拟合
    # 初始参数猜测: [τ=1.0]
    popt_V, pcov_V = fit_model(t_V, V_data, V_model, p0=[1.0])
    perr_V = np.sqrt(np.diag(pcov_V))  # 计算参数的标准误差
    print(f"V(t)模型拟合参数: τ = {popt_V[0]:.3f} ± {perr_V[0]:.3f}") 
    
    # W(t)模型拟合
    # 初始参数猜测: [A=1.0, τ=1.0]
    popt_W, pcov_W = fit_model(t_W, W_data, W_model, p0=[1.0, 1.0])
    perr_W = np.sqrt(np.diag(pcov_W))  # 计算参数的标准误差
    print(f"W(t)模型拟合参数: A = {popt_W[0]:.3f} ± {perr_W[0]:.3f}, τ = {popt_W[1]:.3f} ± {perr_W[1]:.3f}")    # 结果可视化
    plot_results(t_V, V_data, V_model, popt_V, 'V(t) Model Fit')  # 绘制V模型拟合结果
    plot_results(t_W, W_data, W_model, popt_W, 'W(t) Model Fit')  # 绘制W模型拟合结果


