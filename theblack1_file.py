import numpy as np
import os


class TB1Flie():
    def __init__(self):
        return
    
    # 递归创建文件夹
    def create_file(self, path):
        # 递归创建文件夹
        if not os.path.exists(path):
            os.makedirs(path)
    
    # 检测文件编码格式
    def check_charset(self, str_file_path):
        import chardet
        with open(str_file_path, "rb") as f:
            data = f.read(4)
            charset = chardet.detect(data)['encoding']
        
        if charset == "ascii":
            charset = "utf-8"
        return charset

    # 按行读取类txt文件并返回array
    # "split_str"指定分割字符串
    # "start_idx"指定开始行
    def read_by_line(self, str_file_path, split_str = "", start_idx = 1, _read_in_lists = False):
        file_list = []
        __is_read_start = False
        with open(str_file_path, "r" ,encoding=self.check_charset(str_file_path)) as str_file:
            # 读取第一行
            line = str_file.readline()
            line_idx = 1
            #　如果逐行读取，初始化数组
            if _read_in_lists and split_str:
                if split_str == -1:
                    line_data_0 = line.split()
                    NCol = len(line_data_0)
                else:
                    line_data_0 = line.split(split_str)
                    NCol = len(line_data_0)
                
                for _ in range(NCol):
                    file_list.append([])
                    
            # 逐行读取
            while line:
                if __is_read_start:
                    # 是否分割
                    if split_str == -1:
                        line_data = line.split()
                        line_data = np.array(line_data)
                    elif split_str:
                        line_data = line.split(split_str)
                        line_data = np.array(line_data)
                    else:
                        line_data = line
                    # 写入数据
                    if _read_in_lists and split_str:
                        for col_idx in range(NCol):
                            file_list[col_idx].append(line_data[col_idx])
                    else:
                        file_list.append(line_data) # 列表增加
                else:
                    # 是否读取到指定行
                    if line_idx == start_idx:
                        __is_read_start = True
                
                # 读下一行
                line = str_file.readline() # 读取下一行
                line_idx += 1
        
        file_arr = np.array(file_list)
        return file_arr
    
    