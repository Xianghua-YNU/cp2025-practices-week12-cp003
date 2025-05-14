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
    data = np.loadtxt(file_path)
    z = data[:, 0]
    mu = data[:, 1]
    mu_err = data[:, 2]
    return z, mu, mu_err

def hubble_model(z, H0):
    """
    哈勃模型：mu = 5 * log10(cz/H0) + 25
    """
    c = 299792.458  # 光速，单位 km/s
    d_L = (c * z) / H0  # 光度距离（忽略宇宙学常数）
    mu = 5 * np.log10(d_L) + 25
    return mu

def hubble_model_with_deceleration(z, H0, a1):
    """
    包含减速参数的模型：mu ≈ 5 * log10(cz/H0 * [1 + 0.5*(1 - q0)*z]) + 25
    其中 a1 = 0.5*(1 - q0)
    """
    c = 299792.458
    d_L = (c * z / H0) * (1 + a1 * z)
    mu = 5 * np.log10(d_L) + 25
    return mu

def hubble_fit(z, mu, mu_err):
    """
    使用curve_fit拟合哈勃常数
    """
    popt, pcov = curve_fit(hubble_model, z, mu, sigma=mu_err, absolute_sigma=True)
    H0 = popt[0]
    H0_err = np.sqrt(np.diag(pcov))[0]
    return H0, H0_err

def hubble_fit_with_deceleration(z, mu, mu_err):
    """
    拟合哈勃常数和a1参数
    """
    popt, pcov = curve_fit(hubble_model_with_deceleration, z, mu, sigma=mu_err, absolute_sigma=True)
    H0, a1 = popt
    H0_err, a1_err = np.sqrt(np.diag(pcov))
    return H0, H0_err, a1, a1_err

def plot_hubble_diagram(z, mu, mu_err, H0):
    """
    绘制基础哈勃图
    """
    z_fit = np.linspace(min(z), max(z), 500)
    mu_fit = hubble_model(z_fit, H0)

    fig, ax = plt.subplots()
    ax.errorbar(z, mu, yerr=mu_err, fmt='o', label='超新星数据')
    ax.plot(z_fit, mu_fit, 'r-', label=f'拟合模型\nH0 = {H0:.2f} km/s/Mpc')
    ax.set_xlabel("红移 z")
    ax.set_ylabel("距离模数 μ")
    ax.set_title("哈勃图")
    ax.legend()
    return fig

def plot_hubble_diagram_with_deceleration(z, mu, mu_err, H0, a1):
    """
    绘制包含减速参数的哈勃图
    """
    z_fit = np.linspace(min(z), max(z), 500)
    mu_fit = hubble_model_with_deceleration(z_fit, H0, a1)

    fig, ax = plt.subplots()
    ax.errorbar(z, mu, yerr=mu_err, fmt='o', label='超新星数据')
    ax.plot(z_fit, mu_fit, 'g--', label=f'减速模型\nH0 = {H0:.2f}, a1 = {a1:.2f}')
    ax.set_xlabel("红移 z")
    ax.set_ylabel("距离模数 μ")
    ax.set_title("哈勃图（含减速参数）")
    ax.legend()
    return fig

if __name__ == "__main__":
    data_file = "data/supernova_data.txt"
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
