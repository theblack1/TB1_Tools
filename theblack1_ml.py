import numpy as np
import os
import math

# 导入logging模块

class TB1MachineLearning:
    # 初始化函数
    def __init__(self):
        return
    
    # 模拟退火Simulated Annealing
    # https://zhuanlan.zhihu.com/p/404204692
    def SA(self, aim_func, near_func, input_data_arr, init_param_list = None, _example = False,
        init_T = 100, N_loop = 1000, min_T = 20, drop_mode = "fast"):
        
        # 是否启用案例
        if _example:
            print()
            print(f"正在启动TB1库Simulated Annealing案例")
            def aim_func(input_data_arr, param_list):
                res_error = 0
                for x in input_data_arr:
                    y = 1 + 3 * x + 2 * x* x
                    y0 = param_list[0] + param_list[1] * x + param_list[2] * x * x
                    error = y - y0

                    res_error+= error/len(input_data_arr)
                
                return abs(res_error)
            
            def near_func(T, param_list):
                new_param_list = []
                for p in param_list:
                    noise = T/1000 * np.random.uniform(-1,1)
                    new_param_list.append(p + noise)
                
                return new_param_list
                
            input_data_arr = np.array(range(100))
            init_param_list = [1,2,3]
            
            init_T = 100
            N_loop = 1000
            min_T = 20
            drop_mode = "classic"
        
        # 计算初始值
        init_error = aim_func(input_data_arr, init_param_list)
        print()
        print(f"正在启动TB1库Simulated Annealing方法\n初始参数:{init_param_list}\n初始误差值:{init_error}")
        
        def __metropolis(det_error, k_T):
            res = math.exp(-det_error/k_T)
            return res
            
        print()
        # 初始参数
        # 最优解
        best_param_list = init_param_list
        best_error = init_error
        # 当前数值
        k_param_list = init_param_list
        k_error = init_error
        k_T = init_T
        Ndrop = 0
        # 若还没终止，则不断循环
        while k_T >= min_T:
            # 在温度k_T下的搜索循环
            for _ in range(N_loop):
                # 扰动
                new_param_list = near_func(k_T, k_param_list)
                # 计算
                new_error = aim_func(input_data_arr, new_param_list)
                # 判断是否更新
                if new_error < k_error or (np.random.uniform(-1,1) < __metropolis(new_error - k_error, k_T)):
                    # 更新当前值
                    k_error = new_error
                    k_param_list = new_param_list
                    
                    # 若更新了最优值
                    if k_error < best_error:
                        best_error = new_error
                        best_param_list = new_param_list
                        
            print('\r', f"当前温度:{k_T}/{min_T}; 最小误差：{best_error}; 最优解:{best_param_list}", end='', flush=True)
            # 更新温度
            Ndrop += 1
            if drop_mode == "classic":
                k_T = init_T/math.log10(1 + Ndrop)
            elif drop_mode == "fast":
                k_T = init_T/(1 + Ndrop)
        
        print()
        print(f"最终结果：最小误差：{best_error}; 最优解:{best_param_list}")
        return best_param_list, best_error
            
            
                    
            
            

def __test():
    return 1000

ML_TOOL = TB1MachineLearning()
ML_TOOL.SA(aim_func=__test, near_func = __test,input_data_arr=np.array([0,1,-1]), init_param_list= [0,1,3], _example = True)