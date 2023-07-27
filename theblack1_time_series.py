from numpy.fft import fft
import numpy as np
from matplotlib import pyplot as plt
import math
import os

from scipy import signal

class TB1TimeSeries:
    # 初始化函数
    def __init__(self):
        return
    
    # 保存时间序列
    def save_ts_file(self, data_dict, data_name, dir = r"./output/TB1TSTool_default_save", split_word = "\t", _No_subtitle = False):
        # 递归创建文件夹
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        key_arr = np.array(list(data_dict.keys()))
        NData = len(data_dict[key_arr[0]])
        
        with open(dir + "/" + data_name + ".txt", "w") as write_file:
            # 写标题
            if not _No_subtitle:
                title = key_arr[0]
                for key in key_arr[1:]:
                    title = title + split_word + key
                write_file.write(title + '\n')
            
            # 逐行写内容
            for idx in range(NData):
                line = str(data_dict[key_arr[0]][idx])
                for key in key_arr[1:]:
                    line = line + split_word + str((data_dict[key])[idx])
                
                write_file.write(line + "\n")
            
        
        print("成功保存为：" + dir + "/" + data_name + ".txt")
    
    # 快速保存时间序列
    def quick_save_ts(self, t_arr, val_arr, data_name, dir = r"./output/TB1TSTool_default_save", split_word = "\t"):
        # 递归创建文件夹
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        with open(dir + "/" + data_name + ".txt", "w") as write_file:
            # 逐行写内容
            for t, val in zip(t_arr, val_arr):
                line = str(t) + split_word + str(val)
                write_file.write(line + "\n")
            
        
        print("成功保存为：" + dir + "/" + data_name + ".txt")
    
    # 快速读取时间序列
    def quick_read_ts(self, ts_file_path, split_str = ""):
        with open(ts_file_path, "r") as str_file:
            # 读取第一行
            line = str_file.readline()
            
            t_arr = np.array([])
            val_arr = np.array([])
            # 逐行读取
            while line:
                # 是否分割
                if split_str:
                    line_data = line.split(split_str)
                    line_data = np.array(line_data)
                else:
                    line_data = line.split()
                    line_data = np.array(line_data)
                    
                t_arr = np.append(t_arr, line_data[0])
                val_arr = np.append(val_arr, line_data[1])
                
                # 读下一行
                line = str_file.readline() # 读取下一行
        
        return t_arr, val_arr
    
    # 峰值检测
    def find_peak(self, t_arr, val_arr, mode = 1, 
                  peak_lim = None, **kwargs):
        from scipy.signal import argrelextrema
        
        # 是否过滤数据
        if peak_lim:
            if mode == 1:
                val_lim = [peak_lim, np.max(val_arr) + 1]
            elif mode == 0:
                val_lim = [np.min(val_arr) - 1, peak_lim]
        else:
            val_lim = [np.min(val_arr) - 1, np.max(val_arr) + 1]
        
        if mode == 1:
            peak_t_arr = t_arr[argrelextrema(val_arr, np.greater)]
            peak_arr = val_arr[argrelextrema(val_arr, np.greater)]
        elif mode == 0:
            peak_t_arr = t_arr[argrelextrema(val_arr, np.less)]
            peak_arr = val_arr[argrelextrema(val_arr, np.less)]
        
        # 取范围
        peak_t_arr, peak_arr = self.clip_by_val(t_arr=peak_t_arr, val_arr=peak_arr, val_lim=val_lim)
        
        return peak_t_arr, peak_arr
    
    # def normalize(self, val_arr, **kwargs):
        
    # 标准化-最大最小化 
    def min_max_normalize(self, val_arr, min_ = 0, max_ = 0):
        val_max = np.max(val_arr)
        val_min = np.min(val_arr)
        
        # 计算0~1之间的数值
        normalize_val_arr = (val_arr-val_min)/(val_max-val_min)
        
        # 是否有给定最大最小值
        if not (max_ == min_):
            normalize_val_arr = normalize_val_arr * (max_ - min_) + min_
        
        return normalize_val_arr

    # 标准化-按照均值极限
    def avg_lim_normalize(self, val_arr, avg_, lim_list):
        min_ = lim_list[0]
        max_ = lim_list[1]
        
        val_min = np.min(val_arr)
        val_max = np.max(val_arr)
        val_avg = np.average(val_arr)
        
        # a*val_avg + b = avg_
        # a*val_max + b = max_
        # a = (avg_ - max_)/(val_avg - val_max)
        # b = avg_ - a * val_avg
        
        # 极值纠正
        if (val_max - val_avg)/(val_avg - val_min) > (max_ - avg_)/(min_ - avg_):
            # 最大值比最小值离平均值更远
            # 纠正最大值
            a = (avg_ - max_)/(val_avg - val_max)
            b = avg_ - a * val_avg
            
            normalize_val_arr = val_arr * a + b
        else:
            # 最小值比最大值离平均值更远
            # 纠正最小值
            a = (avg_ - min_)/(val_avg - val_min)
            b = avg_ - a * val_avg
            
            normalize_val_arr = val_arr * a + b
        
        return normalize_val_arr
    
    # 序列切割(按照时间)
    def clip_by_t(self, t_arr, val_arr, t_lim):
        # 先统一排序
        t_arr, val_arr = self.sort_by_time(t_arr, val_arr)
        # 取范围
        down_lim = np.min(t_lim)
        up_lim = np.max(t_lim)
        # 切割
        clip_idxs = np.where(np.logical_and(t_arr > down_lim, t_arr < up_lim))
        cliped_t_arr = t_arr[clip_idxs]
        cliped_val_arr = val_arr[clip_idxs]
        
        return cliped_t_arr, cliped_val_arr
    
    # 序列切割(按数值)
    def clip_by_val(self, t_arr, val_arr, val_lim):
        # 取范围
        down_lim = np.min(val_lim)
        up_lim = np.max(val_lim)
        # 切割
        clip_idxs = np.where(np.logical_and(val_arr > down_lim, val_arr < up_lim))
        cliped_t_arr = t_arr[clip_idxs]
        cliped_val_arr = val_arr[clip_idxs]
        
        return cliped_t_arr, cliped_val_arr
    
    # 按照时间重新排序
    def sort_by_time(self, t_arr, val_arr):
        sorted_t_arr = np.sort(t_arr)
        sorted_idx = np.argsort(t_arr)
        
        sorted_val_arr = val_arr[sorted_idx]
        
        return sorted_t_arr, sorted_val_arr
    
    # 时间序列去均值
    def demean(self, val_arr):
        val_avg = np.average(val_arr)
        val_arr = val_arr - val_avg
        
        return val_arr
    
    # 时间序列去趋势
    def detrend(self, t_arr, val_arr, _isShow = False, _example = False):
        # 展示示例
        if _example:
            t_arr=[1,2,3,1.5,4,2.5,6,4,3,5.5,5,2]
            val_arr=[3,4,8,4.5,10,5,15,9,5,16,13,3]
            
            _isShow = True

        # 数据规范化
        t_arr = np.array(t_arr)
        val_arr = np.array(val_arr)
        #　排序以绘制折线图
        t_arr, val_arr = self.sort_by_time(t_arr, val_arr)
        
        # 去趋势
        detrened_val_arr = signal.detrend(val_arr)
        
        if _isShow:
            plt.plot(t_arr, val_arr,color="blue", alpha=0.7)
            plt.plot(t_arr, detrened_val_arr, color="red", alpha=0.7)
            plt.show()
        
        return detrened_val_arr
    
    # 傅里叶变换
    # _get_origin 意思是是否要绘制或者保存原图,True表示同意
    def fft_func(self, Fs, val_arr, _isSave = False, _isShow = False, data_name="",
                _get_origin = True,  _isDemean = False, x_lim = None,
                _example = False, **kwargs):
        # 展示案例
        if _example == True:
            Fs = 10000
            f1 = 400
            f2 = 2e3
            data_name = "TB1 fft_func example data"
            
            _isShow = True
            _isSave = False
            _get_origin = True
            _isDemean = True
            
            # 生成时间序列案例
            start_t = -10
            end_t = 10
            t_arr = np.linspace(start_t, end_t, int(abs((end_t-start_t)*Fs)))
            val_arr = 2 * np.sin(2 * np.pi * f1 * t_arr) + 5 * np.sin(2 * np.pi * f2 * t_arr)
            
            # 加噪
            noise_arr = np.random.normal(start_t, end_t, int(abs((end_t-start_t)*Fs)))
            val_arr = val_arr + noise_arr
        
        # 傅里叶变换
        # 去基波分量
        if _isDemean:
            data_name = data_name + "(Demean)"
            val_arr = self.demean(val_arr)
        
        if _isSave:
            from theblack1_img import TB1Img
            img_tool = TB1Img()
            
        len_ = len(val_arr)
        n = int(np.power(2, np.ceil(np.log2(len_))))
        
        fft_arr = (fft(val_arr, n)) / len_ * 2
        fre_arr = np.arange(int(n / 2)) * Fs / n
        fft_arr = fft_arr[range(int(n / 2))]
        
        # 绘图
            # 是否绘制原图
        if _get_origin:
            plt.figure()
            plt.title(data_name + " time series")
            plt.plot(val_arr)
            plt.grid()
            
            if _isSave:
                plt.savefig(img_tool.get_save_path(file_name=data_name + "_time_series", **kwargs))

        plt.figure()
        plt.title(data_name + " fft result")
        plt.plot(fre_arr, abs(fft_arr))
        # plt.autoscale(enable=True, axis="both", tight=True)
        if x_lim:
            plt.xlim(x_lim[0], x_lim[1])
        plt.grid()
        
        # 是否保存
        if _isSave:
            if "dir" in kwargs.keys():
                plt.savefig(img_tool.get_save_path(file_name=data_name + "_fft_result", **kwargs))
            else:
                plt.savefig(img_tool.get_save_path(file_name=data_name + "_fft_result"))
        
        # 是否显示结果
        if _isShow:
            plt.show()
        
        return fre_arr, abs(fft_arr)
    
    # 时间序列间隔分析
    def t_gap_analysis(self, t_arr, _isShow = True, _hist_bins = 10):
        # 时间间隔提取
        t_gap_list = []
        for t_idx in range(1,len(t_arr)):
            t_gap_list.append(t_arr[t_idx] - t_arr[t_idx-1])
        
        # 时间间隔
        t_gap_arr = np.array(t_gap_list)
        # 原始分布
        if _isShow:
            plt.figure()
            plt.title("time gap original")
            plt.plot(t_gap_arr)
            plt.grid()
            plt.show()
        
        # 排序后分布
        t_gap_arr_sorted = np.sort(t_gap_arr)
        if _isShow:        
            plt.figure()
            plt.title("time gap sorted")
            plt.plot(t_gap_arr_sorted)
            plt.grid()
            plt.show()
        
        # 取独有值分布
        t_gap_arr_unique = np.unique(t_gap_arr_sorted)
        if _isShow:
            plt.figure()
            plt.title("time gap unique")
            plt.plot(t_gap_arr_unique)
            plt.grid()
            plt.show()
        
        # 直方图统计
        if _isShow:
            plt.figure()
            plt.title("time gap hist")
            plt.hist(t_gap_arr, bins=_hist_bins, facecolor="blue", edgecolor="black", alpha=0.7)
            plt.grid()
            plt.show()

        
        return t_gap_arr
    
    # 时间序列重采样
    def resample(self, t_arr, val_arr, resample_rate,
                t_lim = None, **kwargs):
        if t_lim:
            start_t = np.min(t_lim)
            end_t = math.floor((np.max(t_lim) - start_t)/resample_rate)*resample_rate + start_t
        else:
            # 插值范围
            start_t = t_arr[0]
            end_t = math.floor((t_arr[-1] - start_t)/resample_rate)*resample_rate + start_t
            
        #　将结束时间纠正到原序列最大值内
        while end_t > np.max(t_arr):
            end_t -= resample_rate
            
        # 重采样后时间列表
        resample_t_arr = np.array([])
        # 开始重采样
        resample_val_arr = np.array([])
        cur_t = start_t
        
        while cur_t <= end_t:
            resample_t_arr = np.append(resample_t_arr, cur_t)
            
            # 情况1：+-百分之一采样率里有值，对值取平均
            up_lim = cur_t + resample_rate/100
            down_lim = cur_t - resample_rate/100
            where_res1 = np.where(np.logical_and(t_arr > down_lim, t_arr < up_lim))
            if len(where_res1[0]):
                val_res1 = val_arr[where_res1]
                resample_val = np.average(val_res1)
                resample_val_arr = np.append(resample_val_arr, resample_val)
                
                # 向下迭代
                cur_t += resample_rate
                continue
            # 情况2：+-1/2采样率里有值，取加权平均
            up_lim = cur_t + resample_rate/2
            down_lim = cur_t - resample_rate/2
            where_res2 = np.where(np.logical_and(t_arr > down_lim, t_arr < up_lim))
            if len(where_res2[0]):
                val_res2 = val_arr[where_res2]
                resample_val = np.average(val_res2, weights=tuple(abs((t_arr[where_res2]-cur_t))))
                resample_val_arr = np.append(resample_val_arr, resample_val)
                
                # 向下迭代
                cur_t += resample_rate
                continue
            # 情况3：超过+-1/2采样率里有值，取最近的前后两点的加权平均
            search_arr = t_arr - cur_t
            # 求小最近
            down_near_t_idx = np.argmin(abs(search_arr[np.where(search_arr<0)]))
            up_near_t_idx = len(search_arr[np.where(search_arr<0)]) + np.argmin(abs(search_arr[np.where(search_arr>0)]))
            down_near_t = t_arr[down_near_t_idx]
            up_near_t = t_arr[up_near_t_idx]
            down_near_val = val_arr[down_near_t_idx]
            up_near_val = val_arr[up_near_t_idx]
            
            resample_val = np.average(np.array([down_near_val, up_near_val]), weights=tuple(abs(np.array([down_near_t, up_near_t])-cur_t)))
            resample_val_arr = np.append(resample_val_arr, resample_val)
            
            # 向下迭代
            cur_t += resample_rate
            
        return resample_t_arr, resample_val_arr
        
    # 等间隔计算
    def interval_cal(self, t_arr, val_arr, t_interval, core_func):
        t_arr_max = np.max(t_arr)

        res_list = []
        res_t_list = []
        for batch_start in np.arange(0, t_arr_max, t_interval):
            t_lim = [batch_start, batch_start + t_interval]
            # 筛选区间
            cliped_t_arr, cliped_val_arr = self.clip_by_t(t_arr, val_arr, t_lim)
            
            # 计算和记录
            res = core_func(cliped_t_arr, cliped_val_arr)
            res_list.append(res)
            
            res_t = np.average(cliped_t_arr)
            res_t_list.append(res_t)


        res_val_arr = np.array(res_list)
        res_t_arr = np.array(res_t_list)
        
        return res_t_arr, res_val_arr