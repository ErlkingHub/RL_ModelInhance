import numpy as np
from tqdm import tqdm
import scipy.io as sio

MAT_FILE_PATH = ".\\MKP_dataset\\Benchmarks_Chu.mat"

def read_mat(filename):
    '''
    解析指定的 .mat 文件中的所有实例，并返回该实例的字典表示。
    '''
    print(f"Loading {filename}...")
    try:
        if np is None:
            raise ModuleNotFoundError("numpy is required to read .mat benchmark files")
        if sio is None:
            raise ModuleNotFoundError("scipy is required to read .mat benchmark files")
        mat_data = sio.loadmat(filename)
        matrix = mat_data['Problem']
        data_list = []
        total_instances = matrix.shape[0] if matrix.ndim > 0 else matrix.size
        for i in range(total_instances): 
            current_row = matrix[i][0] 
            instance = {
                "id": i + 1,
                "M": int(current_row[0][0][0]),
                "N": int(current_row[1][0][0]),
                "P": np.array(current_row[4], dtype=np.float64), 
                "R": np.array(current_row[5], dtype=np.float64), 
                "B": np.array(current_row[6], dtype=np.float64), 
                "best": float(current_row[9][0][0]),             
            }
            data_list.append(instance)
        return data_list
    except Exception as e:
        print(f"Error: {e}")
        return []


def read_GK(filename):
    '''
    解析 GK .mat 文件中的所有实例，返回与 read_mat 相同格式的字典列表。
    例如：read_GK("MKP_dataset/Benchmarks_GK.mat")。
    '''
    print(f"Loading {filename}...")
    try:
        if np is None:
            raise ModuleNotFoundError("numpy is required to read .mat benchmark files")
        if sio is None:
            raise ModuleNotFoundError("scipy is required to read .mat benchmark files")
        mat_data = sio.loadmat(filename)
        matrix = mat_data['Problem']
        data_list = []
        # GK 与 Chu 的字段顺序不同，按字段名读取以避免错位。
        for i, current_row in enumerate(matrix.flat):
            instance = {
                "id": i + 1,
                "M": int(current_row['m'].item()),
                "N": int(current_row['n'].item()),
                "P": np.array(current_row['P'], dtype=np.float64),
                "R": np.array(current_row['R'], dtype=np.float64),
                "B": np.array(current_row['B'], dtype=np.float64),
                "best": float(current_row['best'].item()),
            }
            data_list.append(instance)
        return data_list
    except Exception as e:
        print(f"Error: {e}")
        return []


def read_mat_specified(filename, index = 0):
    '''
    解析指定的 .mat 文件中的单个实例，并返回该实例的字典表示。
    '''
    print(f"Loading {filename}...")
    try:
        if np is None:
            raise ModuleNotFoundError("numpy is required to read .mat benchmark files")
        if sio is None:
            raise ModuleNotFoundError("scipy is required to read .mat benchmark files")
        mat_data = sio.loadmat(filename)
        matrix = mat_data['Problem']
        data_list = []
        total_instances = matrix.shape[0] if matrix.ndim > 0 else matrix.size
        for i in range(total_instances): 
            current_row = matrix[i][0] 
            instance = {
                "id": i + 1,
                "M": int(current_row[0][0][0]),
                "N": int(current_row[1][0][0]),
                "P": np.array(current_row[4], dtype=np.float64), 
                "R": np.array(current_row[5], dtype=np.float64), 
                "B": np.array(current_row[6], dtype=np.float64), 
                "best": float(current_row[9][0][0]),             
            }
            data_list.append(instance)
        return data_list[index]
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":

    all_instances = read_mat(MAT_FILE_PATH)

    if all_instances:
        
        # 强制指定测试 target_m * target_m 规模的实例
        target_m = 10
        target_n = 100
        
        filtered_instances = [inst for inst in all_instances if inst['M'] == target_m and inst['N'] == target_n]
        print(f"Total instances loaded: {len(all_instances)}")
        print(filtered_instances)
        if filtered_instances:
            print(f"\n[Data Load] 跨规模测试：仅测试 N={target_n}, M={target_m} 的 {len(filtered_instances)} 个实例。")

    # 解析all_instances中的每个实例
    for idx, instance in enumerate(tqdm(all_instances)):
        m, n = instance['M'], instance['N']
        raw_weights, raw_values = instance['R'], instance['P']
        raw_caps = instance['B'].flatten()
        opt_known = instance['best']
        raw_values_flat = raw_values.flatten()
        raw_caps_col = instance['B'] 
        
        scale_w = np.max(raw_weights) + 1e-5
        scale_v = np.max(raw_values) + 1e-5
