import h5py
import os

def split_h5_file(input_h5_path, test_ids_csv_path, output_h5_path1, output_h5_path2):
    """
    Splits an HDF5 file into two based on a list of test IDs.

    Args:
        input_h5_path (str): Path to the input HDF5 file (e.g., 'test.h5').
        test_ids_csv_path (str): Path to the CSV file containing test IDs (e.g., 'rest_test_id.csv').
        output_h5_path1 (str): Path for the first output HDF5 file (contains non-test samples).
        output_h5_path2 (str): Path for the second output HDF5 file (contains test samples).
    """
    try:
        # 1. 读取留出测试集的样本 ID
        test_sample_ids = set()
        with open(test_ids_csv_path, 'r') as f:
            for line in f:
                test_sample_ids.add(line.strip())
        
        print(f"从 {test_ids_csv_path} 中加载 {len(test_sample_ids)} 个测试样本 ID。")

        # 2. 打开输入的 HDF5 文件和两个输出的 HDF5 文件
        with h5py.File(input_h5_path, 'r') as h5_in, \
             h5py.File(output_h5_path1, 'w') as h5_out1, \
             h5py.File(output_h5_path2, 'w') as h5_out2:

            print(f"正在处理输入文件: {input_h5_path}")
            
            num_total_samples = 0
            num_test_samples_found = 0
            num_other_samples_found = 0

            for group_name in h5_in.keys():
                num_total_samples += 1
                if group_name in test_sample_ids:
                    # 这是留出测试集的样本，复制到 test2.h5
                    h5_in.copy(group_name, h5_out2)
                    num_test_samples_found +=1
                else:
                    # 这是其他样本，复制到 test1.h5
                    h5_in.copy(group_name, h5_out1)
                    num_other_samples_found +=1
            
            print(f"总共处理 {num_total_samples} 个样本。")
            print(f"已将 {num_test_samples_found} 个样本写入 {output_h5_path2}。")
            print(f"已将 {num_other_samples_found} 个样本写入 {output_h5_path1}。")

        print("文件分割完成。")

    except FileNotFoundError:
        print(f"错误: 文件未找到。请检查路径 '{input_h5_path}' 或 '{test_ids_csv_path}' 是否正确。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == '__main__':
    # 定义文件路径
    # 请根据您的实际文件位置修改这些路径
    base_dir = "/xiongjun/test/NEW-MIL-dir-backup/backup/NEW-MIL-dir-backup/data/dataset/tcrensuredoinfo5_results_NewRank_KIC_h5_data"
    
    input_h5_file = os.path.join(base_dir, 'test.h5')
    test_ids_file = os.path.join(base_dir, 'rest_test_id.csv')
    output_h5_file1 = os.path.join(base_dir, 'test1.h5') # 包含其余样本
    output_h5_file2 = os.path.join(base_dir, 'test2.h5') # 包含留出测试集样本

    split_h5_file(input_h5_file, test_ids_file, output_h5_file1, output_h5_file2)