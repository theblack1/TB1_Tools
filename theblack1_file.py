import numpy as np
import os


class TB1Flie():
    def __init__(self):
        return
    
    # 递归创建文件夹路径
    def create_file(self, dir):
        # 递归创建文件夹
        if not os.path.exists(dir):
            os.makedirs(dir)
    
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
    # "_read_by_cols"表示读取结果按照每个列保存为数组
    def read_by_line(self, str_file_path, split_str = "", start_idx = 1, _read_by_cols = False):
        file_list = []
        
        with open(str_file_path, "r" ,encoding=self.check_charset(str_file_path)) as str_file:
            # 读取第一行
            line = str_file.readline()
            line = line.replace("\n", "")
            line_idx = 1
            #　如果逐行读取，初始化数组
            if _read_by_cols and split_str:
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
                if line_idx>=start_idx:
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
                    if _read_by_cols and split_str:
                        for col_idx in range(NCol):
                            file_list[col_idx].append(line_data[col_idx])
                    else:
                        file_list.append(line_data) # 列表增加
                
                # 读下一行
                line = str_file.readline() # 读取下一行
                line = line.replace("\n", "")
                line_idx += 1
        
        file_arr = np.array(file_list)
        return file_arr
    