import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# 设置支持中文的字体（如 SimHei 黑体 或 Microsoft YaHei 微软雅黑）
plt.rcParams['font.family'] = 'SimHei'  # 或 'Microsoft YaHei'

# 正确显示负号
plt.rcParams['axes.unicode_minus'] = False

def load_supernova_data(file_path):
    """
    从文件中加载超新星数据
    """
   # 使用numpy.loadtxt加载CSV文件
    data = np.loadtxt(file_path, delimiter='\t', skiprows=6)
    z = data[:, 0]       # 第一列：红移
    mu = data[:, 1]      # 第二列：距离模数
    mu_err = data[:, 2]  # 第三列：距离模数误差
    return z, mu, mu_err

def hubble_model(z, H0):
    """
    哈勃模型：mu = 5 * log10(cz/H0) + 25
    """
    c = 299792.458  # 光速，单位 km/s
    mu = 5 * np.log10(c * z / H0) + 25
    return mu

def hubble_model_with_deceleration(z, H0, a1):
    """
    包含减速参数的模型：mu ≈ 5 * log10(cz/H0 * [1 + 0.5*(1 - q0)*z]) + 25
    其中 a1 = 0.5*(1 - q0)
    """
    c = 299792.458
    mu = 5 * np.log10(c * z / H0 * (1 + 0.5 * (1 - a1) * z)) + 25
    return mu

def hubble_fit(z, mu, mu_err):
    """
    使用curve_fit拟合哈勃常数
    """
    H0_guess = 70.0  # km/s/Mpc
    popt, pcov = curve_fit(hubble_model, z, mu, p0=[H0_guess], sigma=mu_err, absolute_sigma=True)
    H0 = popt[0]
    H0_err = np.sqrt(pcov[0,0])
    return H0, H0_err

def hubble_fit_with_deceleration(z, mu, mu_err):
    """
    拟合哈勃常数和a1参数
    """
   # 初始猜测值
    H0_guess = 70.0  # km/s/Mpc
    a1_guess = 1.0   # 对应于q0=0
    
    # 使用curve_fit进行加权最小二乘拟合
    popt, pcov = curve_fit(hubble_model_with_deceleration, z, mu, 
                          p0=[H0_guess, a1_guess], sigma=mu_err, absolute_sigma=True)
    
    # 从拟合结果中提取参数及其误差
    H0 = popt[0]
    a1 = popt[1]
    H0_err = np.sqrt(pcov[0, 0])
    a1_err = np.sqrt(pcov[1, 1])
    return H0, H0_err, a1, a1_err

def plot_hubble_diagram(z, mu, mu_err, H0):
    """
    绘制基础哈勃图
    """
     # 创建一个新的图形
    plt.figure(figsize=(10, 6))
    
    # 绘制数据点（带误差棒）
    plt.errorbar(z, mu, yerr=mu_err, fmt='o', color='blue', markersize=5, 
                 ecolor='gray', elinewidth=1, capsize=2, label='Supernova data')
    z_fit = np.linspace(min(z), max(z), 1000)
    mu_fit = hubble_model(z_fit, H0)

    # 绘制最佳拟合曲线
    plt.plot(z_fit, mu_fit, '-', color='red', linewidth=2, 
             label=f'Best fit: $H_0$ = {H0:.1f} km/s/Mpc')
    
    # 添加轴标签和标题
    plt.xlabel('Redshift z')
    plt.ylabel('Distance modulus μ')
    plt.title('Hubble Diagram')
    
    # 添加图例
    plt.legend()
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    # 调整布局
    plt.tight_layout()
    return plt.gcf()

def plot_hubble_diagram_with_deceleration(z, mu, mu_err, H0, a1):
    """
    绘制包含减速参数的哈勃图
    """
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))
    
    # 绘制数据点（带误差棒）
    plt.errorbar(z, mu, yerr=mu_err, fmt='o', color='blue', markersize=5, 
                 ecolor='gray', elinewidth=1, capsize=2, label='Supernova data')
    
    # 生成用于绘制拟合曲线的红移值（更密集）
    z_fit = np.linspace(min(z), max(z), 1000)
    
    # 计算拟合曲线上的距离模数值
    mu_fit = hubble_model_with_deceleration(z_fit, H0, a1)
    # 绘制最佳拟合曲线
    plt.plot(z_fit, mu_fit, '-', color='red', linewidth=2, 
             label=f'Best fit: $H_0$ = {H0:.1f} km/s/Mpc, $a_1$ = {a1:.2f}')
    # 添加轴标签和标题
    plt.xlabel('Redshift z')
    plt.ylabel('Distance modulus μ')
    plt.title('Hubble Diagram with Deceleration Parameter')
    # 添加图例
    plt.legend()
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    # 调整布局
    plt.tight_layout()
    return plt.gcf()

if __name__ == "__main__":
    data_file = " /home/runner/work/cp2025-practices-week12-cp003/cp2025-practices-week12-cp003/Supernova_Hubble_Fitting/data/supernova_data.txt"
    z, mu, mu_err = load_supernova_data(data_file)

    H0, H0_err = hubble_fit(z, mu, mu_err)
    print(f"拟合得到的哈勃常数: H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")
    fig = plot_hubble_diagram(z, mu, mu_err, H0)
    plt.show()

    H0, H0_err, a1, a1_err = hubble_fit_with_deceleration(z, mu, mu_err)
    print(f"拟合得到的哈勃常数: H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")
    print(f"拟合得到的a1参数: a1 = {a1:.2f} ± {a1_err:.2f}")
    fig = plot_hubble_diagram_with_deceleration(z, mu, mu_err, H0, a1)
    plt.show()
