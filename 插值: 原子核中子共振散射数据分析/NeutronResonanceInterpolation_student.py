import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 设置中文字体（优化多系统兼容）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False

# 实验数据
energy = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])  # MeV
cross_section = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])  # mb
error = np.array([9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14])  # mb

def lagrange_interpolation(x, x_data, y_data):
    """拉格朗日多项式插值"""
    result = 0.0
    for i in range(len(x_data)):
        term = y_data[i]
        for j in range(len(x_data)):
            if j != i:
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term
    return result

def cubic_spline_interpolation(x, x_data, y_data):
    """三次样条插值"""
    spline = interp1d(x_data, y_data, kind='cubic', fill_value='extrapolate')
    return spline(x)

def find_peak(x, y):
    """寻找峰值位置和半高全宽(FWHM)"""
    peak_idx = np.argmax(y)
    peak_x = x[peak_idx]
    peak_y = y[peak_idx]
    
    # 计算半高全宽
    half_max = peak_y / 2
    left_idx = np.argmin(np.abs(y[:peak_idx] - half_max))
    right_idx = peak_idx + np.argmin(np.abs(y[peak_idx:] - half_max))
    fwhm = x[right_idx] - x[left_idx]
    
    return peak_x, fwhm

def plot_results():
    """绘制插值结果和原始数据对比图"""
    x_interp = np.linspace(0, 200, 500)  # 密集插值点
    
    # 计算插值结果
    lagrange_result = np.array([lagrange_interpolation(x, energy, cross_section) for x in x_interp])
    spline_result = cubic_spline_interpolation(x_interp, energy, cross_section)
    
    # 计算峰值和FWHM
    lagrange_peak, lagrange_fwhm = find_peak(x_interp, lagrange_result)
    spline_peak, spline_fwhm = find_peak(x_interp, spline_result)
    
    # 绘图
    plt.figure(figsize=(12, 6))
    plt.errorbar(energy, cross_section, yerr=error, fmt='o', color='black', label='实验数据', capsize=5)
    plt.plot(x_interp, lagrange_result, '-', label=f'拉格朗日插值 (峰值: {lagrange_peak:.2f} MeV)')
    plt.plot(x_interp, spline_result, '--', label=f'三次样条插值 (峰值: {spline_peak:.2f} MeV)')
    
    # 标记峰值位置
    plt.axvline(lagrange_peak, color='blue', linestyle=':', alpha=0.5)
    plt.axvline(spline_peak, color='orange', linestyle=':', alpha=0.5)
    
    # 图表装饰
    plt.xlabel('能量 (MeV)')
    plt.ylabel('截面 (mb)')
    plt.title('中子共振散射截面分析')
    plt.legend()
    plt.grid(True)
    
    # 打印结果
    print(f"拉格朗日插值 - 峰值位置: {lagrange_peak:.2f} MeV, FWHM: {lagrange_fwhm:.2f} MeV")
    print(f"三次样条插值 - 峰值位置: {spline_peak:.2f} MeV, FWHM: {spline_fwhm:.2f} MeV")
    
    plt.tight_layout()
    plt.savefig('插值对比.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_results()
