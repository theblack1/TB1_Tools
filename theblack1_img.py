from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import cv2
import numpy as np
import os
import time
from threading import Thread
# color map信息请见：https://zhuanlan.zhihu.com/p/114420786

DEFAULT_PLT_COLORMAP = cm.magma
DEFAULT_CV_COLORMAP = cv2.COLORMAP_PLASMA
COLORS=list(mcolors.TABLEAU_COLORS.keys()) # cur_color = mcolors.TABLEAU_COLORS[COLORS[idx]]

class TB1Img():
    # 初始化函数
    def __init__(self):
        return
    
    # 生成保存路径
    def get_save_dir(self, file_name, path = "./figs/TB1ImgTool_default_save", driver = 'png', _time_labled = True):
        import os
        import time
        
        # 递归创建文件夹
        if not os.path.exists(path):
            os.makedirs(path)
        
        # 是否使用时间标签来保证每一次保存都有不同名称
        time_string = ''
        if _time_labled:
            time_string = time.strftime("%Y-%m-%d_%Hh-%Mm-%Ss",time.localtime(time.time()))
        
        # 生成文件名
        file_save_name = path + '/' + time_string + '_' + file_name + '.' + driver
        
        return file_save_name
    
    # 使用opencv展示图像
    def show_in_cv(self, img_array, win_name = 'img',save_name = '', _stretch = False, _hist_enhance = False, _fake_color = DEFAULT_CV_COLORMAP,
                _step = 0.9, _max_zoom=3, _show = True, _each_stretch = False):
        tb1_cv_imshow = _TB1CvImshow(img_array, win_name, save_name, _stretch, _hist_enhance, _fake_color,
                _step, _max_zoom, _show, _each_stretch)
        return  tb1_cv_imshow.output_img_dict
    
    # 对多组一维数据绘制分图
    # mode_arr 表征分图排序，还有叠图次数（按照顺序）,
    # 如mode_arr = array[[0,0],[1,2]]表示两行两列的图，且第二行第一列的图叠1次，第二行第二列叠2次
    def draw_ts(self, data_dict, mode_arr, _isSave = True,  _isShow = True, **kwargs):
        fig = plt.figure(figsize=(12,6))
        # 标准化
        mode_arr = np.array(mode_arr)
        
        # 是否有标题
        if "title" in kwargs.keys():
            fig.title = kwargs["title"]
            file_name = kwargs["title"]
        else:
            file_name = "ts_fig"
            
        # 是否有路径
        if "path" in kwargs.keys():
            path = kwargs["path"]
        else:
            path = "./figs/TB1ImgTool_default_save"
        
        # 是否有透明度数据
        if "alpha" in kwargs.keys():
            alpha = kwargs["alpha"]
        else:
            alpha = 0.5
        
        key_arr = np.array(list(data_dict.keys()))
        
        # 解析mode_arr
        mode_arr_flatten = mode_arr.flatten()
        if len(mode_arr_flatten) == 1:
            NRow = 1
            NCol = 1
        else:
            NRow = mode_arr.shape[0]
            NCol = mode_arr.shape[1]
        
        
        # 逐个遍历
        key_iter = iter(key_arr)
        color_iter = iter(COLORS)
        
        for (fig_idx,mode) in enumerate(mode_arr_flatten):
            # 读取数据
            cur_key = next(key_iter)
            
            data_name = cur_key
            t_arr = data_dict[cur_key][0]
            ts_arr = data_dict[cur_key][1]
            
            data_color = mcolors.TABLEAU_COLORS[next(color_iter)]
            # 绘制
            if mode == 0:
                # 不叠图
                ax = fig.add_subplot(NRow, NCol, fig_idx +1)
                
                # 绘图
                ax.grid()
                line,  = ax.plot(t_arr, ts_arr, color = data_color)
                ax.set_title(data_name)
                
                ax.legend((line),[data_name])
            else:
                # 叠图
                ax = fig.add_subplot(NRow, NCol, fig_idx + 1)
                # 合成标题
                sub_title = data_name
                # 绘图
                ax.grid()
                
                # 准备图例数据
                line_list = []
                data_name_list = []
                line, = ax.plot(t_arr, ts_arr, color = data_color, alpha = alpha)
                line_list.append(line)
                data_name_list.append(data_name)
                
                for _ in range(mode):
                    # 读取其他数据
                    cur_key = next(key_iter)
                    
                    data_name = cur_key
                    t_arr = data_dict[cur_key][0]
                    ts_arr = data_dict[cur_key][1]
                    data_color = mcolors.TABLEAU_COLORS[next(color_iter)]
                    # 合成标题
                    sub_title = sub_title + "_" + data_name
                    # 绘图
                    line, = ax.plot(t_arr, ts_arr, color = data_color)
                    line_list.append(line)
                    data_name_list.append(data_name)
                    
                    ax.set_title(sub_title)
            
            ax.legend(tuple(line_list), data_name_list)
            ax.autoscale(enable=True, axis="x", tight=True)
        
        plt.tight_layout()
        if _isSave:
            plt.savefig(self.get_save_dir(file_name=file_name, path = path))
        
        if _isShow:
            plt.show()
        else:
            return fig
    
    # 随机颜色
    def random_color(self):
        idx = np.random.randint(0, len(COLORS))
        cur_color = mcolors.TABLEAU_COLORS[COLORS[idx]]
        
        return cur_color
    
    # 矩阵转热力图
    def mat_to_heatmap(self, input_mat, file_name = "", cmap=DEFAULT_PLT_COLORMAP):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        #　生成标签
        label = ["{}".format(i) for i in range(1, input_mat.shape[0]+1)]
        df = pd.DataFrame(input_mat, index=label, columns=label)

        # 绘制热力图
        plt.figure(figsize=(7.5, 6.3))
        ax = sns.heatmap(df, xticklabels=df.corr().columns, 
                        yticklabels=df.corr().columns, cmap=cmap,
                        linewidths=6, annot=True)

        # 设置坐标系
        plt.xticks(fontsize=16,family='Times New Roman')
        plt.yticks(fontsize=16,family='Times New Roman')

        # 保存文件
        if len(file_name):
            save_dir = self.get_save_dir(file_name = file_name)
            plt.savefig(save_dir)
        
        # 展示热力图
        plt.tight_layout()
        plt.show()
    
    # 矩阵转3D图
    # 参考代码https://blog.csdn.net/qq_40811682/article/details/117027899
    def mat_to_3D(self, input_mat, img_type, file_name = "", cmap=DEFAULT_PLT_COLORMAP):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np
        
        # 初始化3D图像
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        # 准备数据
        X = range(input_mat.shape[0])
        Y = range(input_mat.shape[1])
        X, Y = np.meshgrid(X, Y)
        Z = input_mat
        
        # 绘制表面
        if img_type == "surface":
            # 表面图Surface plots
            surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                                linewidth=0, antialiased=False)
        elif img_type == "tri-surface":
            surf = ax.plot_trisurf(np.array(X).flatten(), np.array(Y).flatten(), Z.flatten(), cmap = cmap)
            
        
        # 定制化坐标系
        ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        # 添加颜色条
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # 保存文件
        if len(file_name):
            save_dir = self.get_save_dir(file_name = file_name)
            plt.savefig(save_dir)
        
        # 显示图像
        plt.show()
    
    # # 落点热力图
    # def drop_heatmap(self, x_arr, y_arr, map_size, mode = "times"):
    #     import numpy.matlib as nm
    #     from theblack1_time_series import TB1TimeSeries
    #     TS_TOOL = TB1TimeSeries()
    #     # 初始化
    #     Nx = map_size[0]
    #     Ny = map_size[1]
    #     drop_mat = nm.empty((Ny, Nx))
        
    #     # 生成查找方格位置
    #     x_interval = (np.max(x_arr) - np.min(x_arr))/Nx
    #     y_interval = (np.max(y_arr) - np.min(y_arr))/Ny
    #     # 进入x batch
    #     for x_batch_start in np.arange(np.min(x_arr), np.max(x_arr), x_interval):
    #         x_batch_end = x_batch_start + x_interval
    #         # 寻找该x batch 中符合条件的x索引
    #         x_where_res = np.where(
    #             np.logical_and(np.logical_or(x_arr > x_batch_start, x_arr == x_batch_start), x_arr < x_batch_end))
    #         # 在x batch 中寻找y batch
    #         for y_batch_start in np.arange(np.min(y_arr), np.max(y_arr), y_interval):
    #             y_batch_end = y_batch_start + y_interval
    #             # 寻找该x batch 中符合条件的x索引
    #             y_where_res = np.where(np.logical_and(np.logical_or(y_arr > y_batch_start, y_arr == y_batch_start), y_arr < y_batch_end))
                
    #             # 求索引交集
    #             xy_res = x_where_res[np.in1d(x_where_res, y_where_res)]  
                
    # 二维数组化一维数组处理并还原
    # #(部分与二维无关的数据处理（如缠绕），可以这样处理来提高运算速度)
    def flatten_process(self, func, input_data, **param):
        # 2D化1D
        flatten_data = input_data.flatten()
        
        # 处理过程
        flatten_data_result = func(flatten_data, **param)
        
        # 数据恢复2D
        output_data = flatten_data_result.reshape(input_data.shape[0],input_data.shape[1])

        return output_data
    
    # 窗口计算
    def win_process(self, core_func, input_mat, win_mask, slide=[1,1], **params):
        # todo 添加padding功能
        # todo 添加多线程
        # todo 添加多通道
        
        # 初始化数据(统一转化为array，防止报错)
        win_mask = np.array(win_mask)
        input_mat = np.array(input_mat)
        
        # 获取窗口中心坐标
        x_center,y_center = np.where(win_mask == 1)
        x_center = x_center[0]
        y_center = y_center[0]
        
        # 读取窗口长宽,和矩阵长宽
        h_win,w_win = win_mask.shape

        h_mat = input_mat.shape[0]
        w_mat = input_mat.shape[1]
        
        # 计算窗口上下左右边界
        barrier_up = x_center
        barrier_down = h_win - (x_center + 1)
        barrier_left = y_center
        barrier_right = w_win - (y_center + 1)
        
        # 读取步长
        [slide_x, slide_y] = slide
        
        # 计算横向和纵向可能取值范围
        y_range = range(barrier_left, w_mat - barrier_right, slide_y)
        x_range = range(barrier_up, h_mat - barrier_down, slide_x)
        
        # >遍历所有可能窗口
        res_list = []
        center_idxs_list = []
        # 遍历中心点位置
        for i_center in x_range:
            for j_center in y_range:
                # >获取窗口所有数据
                # 初始化
                win_mat = np.zeros_like(win_mask, dtype = np.dtype(np.dtype(input_mat.flatten()[0])))

                # 窗口第一个:[i_center - barrier_up, j_center - barrier_left]
                x_win_first = i_center - barrier_up
                y_win_first = j_center - barrier_left
                # 窗口最后一个:[i_center + barrier_down, j_center + barrier_right]
                x_win_last = i_center + barrier_down
                y_win_last = j_center + barrier_right

                for win_idx_x, i_win in enumerate(range(x_win_first, x_win_last + 1)):
                    for win_idx_y, j_win in enumerate(range(y_win_first, y_win_last + 1)):
                        win_mat[win_idx_x, win_idx_y] = input_mat[i_win, j_win]
                        # print(input_mat[i_win, j_win])
                        
                # 核函数处理
                res_list.append(core_func(win_mat, **params))

                # 记录中心点坐标集
                center_idxs_list.append([i_center, j_center])

        
        return res_list, center_idxs_list

class _TB1CvImshow():
    # 创建类，自动处理并显示
    def __init__(self, img_array, win_name = 'img',save_name = '', _stretch = False, _hist_enhance = False, _fake_color = DEFAULT_CV_COLORMAP,
                _step = 0.9, _max_zoom=3, _show = True, _each_stretch = False):
        # 准备输出的数据
        self.output_img_dict = {}
        
        # 读取数据
        self._hist_enhance = _hist_enhance
        self._fake_color = _fake_color
        self.save_name = save_name
        self._each_stretch = _each_stretch
        
        if _each_stretch:
            _stretch = True
        
        # 自适应窗口宽高
        self.origin_img_wh = np.shape(img_array)
        self.origin_h = self.origin_img_wh[0]
        self.origin_w = self.origin_img_wh[1]
        
        if self.origin_w > self.origin_h:
            # 宽形
            self.g_window_wh = [1920, int(1080*self.origin_h/self.origin_w)]  # 窗口宽高
        else:
            # 高型
            self.g_window_wh = [int(1080*self.origin_w/self.origin_h), 1080]  # 窗口宽高
        
        # 全局变量
        self.g_window_name = win_name  # 窗口名

        self.g_location_win = [0, 0] # 相对于大图，窗口在图片中的位置
        self.location_win = [0, 0]  # 鼠标左键点击时，暂存self.g_location_win
        self.g_location_click, self.g_location_release = [0, 0], [0, 0]  # 相对于窗口，鼠标左键点击和释放的位置

        self.g_zoom, self.g_step = 1, _step  # 图片缩放比例和缩放系数
        self._max_zoom = _max_zoom
        
        self.g_image_original = np.zeros_like(img_array, dtype=np.uint8) # 原始图像
        
        img_array_fix = np.copy(img_array)
        
        # 数据拉伸
        if _stretch:
            if img_array_fix.ndim == 3 and self._each_stretch:
                # 如果多通道图像要各自拉伸
                # 通道拆分
                b_img, g_img, r_img = cv2.split(img_array_fix)
                
                # 各自处理
                cv2.normalize(b_img,b_img,0,255,cv2.NORM_MINMAX)
                cv2.normalize(g_img,g_img,0,255,cv2.NORM_MINMAX)
                cv2.normalize(r_img,r_img,0,255,cv2.NORM_MINMAX)
                
                # 通道合并
                img_array_fix = cv2.merge([b_img, g_img, r_img])
            else:    
                cv2.normalize(img_array,img_array_fix,0,255,cv2.NORM_MINMAX)
            
            # 保存数据
            self.output_img_dict['stretch'] = img_array_fix
            if img_array_fix.ndim == 3:
                # 通道拆分
                b_img, g_img, r_img = cv2.split(img_array_fix)
                self.output_img_dict['stretch_b'] = b_img
                self.output_img_dict['stretch_g'] = g_img
                self.output_img_dict['stretch_r'] = r_img
        
        # 直方图强化或伪彩色合成
        if self._hist_enhance or (self._fake_color and (not img_array.ndim == 3)):
            img_array_fix_enhance = self.enhance(src=img_array_fix)
            self.g_image_original = img_array_fix_enhance
            # 如果准备保存
            if len(self.save_name):
                img_array_fix_toSave = np.copy(img_array)
                cv2.normalize(np.copy(img_array.astype(np.uint8)),img_array_fix_toSave,0,255,cv2.NORM_MINMAX)
                # self.save_result(self.save_name + "_origin", img_array_fix_toSave)
                self.save_result(self.save_name + "(enhanced)", img_array_fix_enhance)
        else:
            self.g_image_original = img_array_fix
            if len(self.save_name):
                img_array_fix_toSave = np.copy(img_array)
                cv2.normalize(np.copy(img_array.astype(np.uint8)),img_array_fix_toSave,0,255,cv2.NORM_MINMAX)
                self.save_result(self.save_name, img_array_fix_toSave)
        
        self.g_image_zoom = self.g_image_original.copy()  # 缩放后的图片
        self.g_image_show = self.g_image_original[self.g_location_win[1]:self.g_location_win[1] + self.g_window_wh[1], self.g_location_win[0]:self.g_location_win[0] + self.g_window_wh[0]]  # 实际显示的图片

        # 显示图像
        if _show:
            self.thread(self.show_img)

    # 数据对比度强化
    def enhance(self, src):
        if self._hist_enhance:
            # 如果图像多通道，则所有通道合并处理
            if src.ndim == 3:
                # 通道拆分
                b_img, g_img, r_img = cv2.split(src)
                
                # 临时组合运算
                hstack_img = np.hstack((b_img, g_img, r_img))
                dst_hstack_img = cv2.equalizeHist(hstack_img.astype(np.uint8))
                
                # 横向拆分
                dst_hsplit = np.hsplit(dst_hstack_img, 3)
                
                dst = cv2.merge([dst_hsplit[0], dst_hsplit[1], dst_hsplit[2]])
                
                # 储存数据
                self.output_img_dict["hist_enhance_b"] = dst_hsplit[0]
                self.output_img_dict["hist_enhance_g"] = dst_hsplit[1]
                self.output_img_dict["hist_enhance_r"] = dst_hsplit[2]
                self.output_img_dict["hist_enhance"] = dst
            else:
                dst = cv2.equalizeHist(src.astype(np.uint8))       #自动调整图像对比度，把图像变得更清晰
        else:
            dst = src.astype(np.uint8)
        
        # 假彩色
        # 如果图像有多个通道，则自动放弃伪彩色
        if src.ndim == 3:
            return dst
        elif self._fake_color:
            dst = cv2.applyColorMap(cv2.convertScaleAbs(dst), self._fake_color)
            
            # 储存数据
            self.output_img_dict["fake_color"] = dst
        
        return dst
        
    # 保存内容
    def save_result(self, file_name, img, driver = 'jpg'):
        path = "./data/imshow_save"
        # 递归创建文件夹
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_string = time.strftime("%Y-%m-%d_%Hh-%Mm-%Ss",time.localtime(time.time()))
        file_save_name = path + '/' + time_string + '_' + file_name + '.' + driver
        # file_save_name = path + '/' + file_name + '.' + driver
        
        cv2.imwrite(file_save_name, img)
    
    # 线程
    def thread(self,func,*args):#fun是一个函数  args是一组参数对象
        '''将函数打包进线程'''
        t=Thread(target=func,args=args)#target接受函数对象  arg接受参数  线程会把这个参数传递给func这个函数
        t.setDaemon(True)#守护
        t.start()#启动线程
        t.join()
    
    # 显示影像
    def show_img(self):
        # 设置窗口
        cv2.namedWindow(self.g_window_name, cv2.WINDOW_NORMAL)
        # 设置窗口大小，只有当图片大于窗口时才能移动图片
        cv2.resizeWindow(self.g_window_name, self.g_window_wh[0], self.g_window_wh[1])
        cv2.moveWindow(self.g_window_name, 0, 0)  # 设置窗口在电脑屏幕中的位置
        # 鼠标事件的回调函数
        cv2.setMouseCallback(self.g_window_name, self.mouse)
        cv2.waitKey()  # 不可缺少，用于刷新图片，等待鼠标操作
        
        cv2.destroyAllWindows()
    
    # 矫正窗口在图片中的位置
    # img_wh:图片的宽高, win_wh:窗口的宽高, win_xy:窗口在图片的位置
    def check_location(self, img_wh, win_wh, win_xy):
        for i in range(2):
            if win_xy[i] < 0:
                win_xy[i] = 0
            elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] > win_wh[i]:
                win_xy[i] = img_wh[i] - win_wh[i]
            elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] < win_wh[i]:
                win_xy[i] = 0
        # print(img_wh, win_wh, win_xy)

    # 计算缩放倍数
    # flag：鼠标滚轮上移或下移的标识, step：缩放系数，滚轮每步缩放0.1, zoom：缩放倍数
    def count_zoom(self,flag, _step, zoom):
        if flag > 0:  # 滚轮上移
            zoom /= _step
            if zoom > self._max_zoom:  # 最多只能放大到1倍
                zoom = self._max_zoom
        else:  # 滚轮下移
            zoom *= _step
            if zoom < 0.01:  # 最多只能缩小到0.01倍
                zoom = 0.01
        zoom = round(zoom, 2)  # 取2位有效数字
        return zoom

    # OpenCV鼠标事件
    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            self.g_location_click = [x, y]  # 左键点击时，鼠标相对于窗口的坐标
            self.location_win = [self.g_location_win[0], self.g_location_win[1]]  # 窗口相对于图片的坐标，不能写成location_win = self.g_location_win
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
            self.g_location_release = [x, y]  # 左键拖曳时，鼠标相对于窗口的坐标
            h1, w1 = self.g_image_zoom.shape[0:2]  # 缩放图片的宽高
            w2, h2 = self.g_window_wh  # 窗口的宽高
            show_wh = [0, 0]  # 实际显示图片的宽高
            if w1 < w2 and h1 < h2:  # 图片的宽高小于窗口宽高，无法移动
                show_wh = [w1, h1]
                self.g_location_win = [0, 0]
            elif w1 >= w2 and h1 < h2:  # 图片的宽度大于窗口的宽度，可左右移动
                show_wh = [w2, h1]
                self.g_location_win[0] =self.location_win[0] + self.g_location_click[0] - self.g_location_release[0]
            elif w1 < w2 and h1 >= h2:  # 图片的高度大于窗口的高度，可上下移动
                show_wh = [w1, h2]
                self.g_location_win[1] =self.location_win[1] + self.g_location_click[1] - self.g_location_release[1]
            else:  # 图片的宽高大于窗口宽高，可左右上下移动
                show_wh = [w2, h2]
                self.g_location_win[0] =self.location_win[0] + self.g_location_click[0] - self.g_location_release[0]
                self.g_location_win[1] =self.location_win[1] + self.g_location_click[1] - self.g_location_release[1]
            self.check_location([w1, h1], [w2, h2], self.g_location_win)  # 矫正窗口在图片中的位置
            self.g_image_show = self.g_image_zoom[self.g_location_win[1]:self.g_location_win[1] + show_wh[1], self.g_location_win[0]:self.g_location_win[0] + show_wh[0]]  # 实际显示的图片
        elif event == cv2.EVENT_MOUSEWHEEL:  # 滚轮
            z = self.g_zoom  # 缩放前的缩放倍数，用于计算缩放后窗口在图片中的位置
            self.g_zoom = self.count_zoom(flags, self.g_step, self.g_zoom)  # 计算缩放倍数
            w1, h1 = [int(self.g_image_original.shape[1] * self.g_zoom), int(self.g_image_original.shape[0] * self.g_zoom)]  # 缩放图片的宽高
            w2, h2 = self.g_window_wh  # 窗口的宽高
            self.g_image_zoom = cv2.resize(self.g_image_original, (w1, h1), interpolation=cv2.INTER_AREA)  # 图片缩放
            show_wh = [0, 0]  # 实际显示图片的宽高
            if w1 < w2 and h1 < h2:  # 缩放后，图片宽高小于窗口宽高
                show_wh = [w1, h1]
                cv2.resizeWindow(self.g_window_name, w1, h1)
            elif w1 >= w2 and h1 < h2:  # 缩放后，图片高度小于窗口高度
                show_wh = [w2, h1]
                cv2.resizeWindow(self.g_window_name, w2, h1)
            elif w1 < w2 and h1 >= h2:  # 缩放后，图片宽度小于窗口宽度
                show_wh = [w1, h2]
                cv2.resizeWindow(self.g_window_name, w1, h2)
            else:  # 缩放后，图片宽高大于窗口宽高
                show_wh = [w2, h2]
                cv2.resizeWindow(self.g_window_name, w2, h2)
                
            self.g_location_win = [int((self.g_location_win[0] + x) * self.g_zoom / z - x), int((self.g_location_win[1] + y) * self.g_zoom / z - y)]  # 缩放后，窗口在图片的位置
            self.check_location([w1, h1], [w2, h2], self.g_location_win)  # 矫正窗口在图片中的位置
            # print(self.g_location_win, show_wh)
            self.g_image_show = self.g_image_zoom[self.g_location_win[1]:self.g_location_win[1] + show_wh[1], self.g_location_win[0]:self.g_location_win[0] + show_wh[0]]  # 实际的显示图片
        
        cv2.imshow(self.g_window_name, self.g_image_show)