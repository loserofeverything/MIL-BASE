import h5py
import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
from esm.models.esmc import ESMC

def batch_sequences(sequences, batch_size=64):
    """将序列列表分成批次"""
    for i in range(0, len(sequences), batch_size):
        yield sequences[i:i + batch_size]

def generate_embeddings(model, sequences, device, batch_size=64):
    """为序列批次生成嵌入"""
    embeddings = []
    
    for batch in batch_sequences(sequences, batch_size):
        with torch.no_grad(), torch.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            input_ids = model._tokenize(batch)
            output = model(input_ids.to(device))
            
            # hiddens = output.hidden_states
            # batch_embeddings = torch.mean(hiddens[-1], dim=1)
            embs = output.embeddings
            batch_embeddings = torch.mean(embs, dim=1)
            
            batch_embeddings = batch_embeddings.to(torch.float32).cpu().detach().numpy()
            
            embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings) if embeddings else np.array([])

def process_h5_file(file_path, device, batch_size=64):
    """处理H5文件，为每个k-mer添加嵌入"""
    print(f"处理文件: {file_path}")
    
    # 加载ESM模型
    print("加载ESM模型...")
    model = ESMC.from_pretrained("esmc_300m")
    model = model.to(device)
    model.eval()
    
    # 打开H5文件进行读写
    with h5py.File(file_path, 'r+') as f:
        # 获取所有样本ID
        sample_ids = list(f.keys())
        print(f"找到 {len(sample_ids)} 个样本")
        
        # 遍历每个样本
        for sample_id in tqdm(sample_ids, desc="处理样本"):
            group = f[sample_id]
            
            # 读取k-mer序列
            kmers = group['kmer'][...]
            # 将字节字符串转换为Python字符串
            kmers = [kmer.decode('utf-8') for kmer in kmers]
            
            # 生成嵌入
            print(f"为样本 {sample_id} 生成 {len(kmers)} 个k-mer的嵌入")
            embeddings = generate_embeddings(model, kmers, device, batch_size)
            
            # 检查嵌入是否已存在，如果存在则删除
            if 'embeddings' in group:
                del group['embeddings']
            
            # 保存嵌入
            group.create_dataset('embeddings', data=embeddings, dtype=np.float32)
            
            print(f"样本 {sample_id} 处理完成，嵌入维度: {embeddings.shape}")
    
    print(f"文件 {file_path} 处理完成")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 处理所有指定的H5文件
    for file_path in args.h5_files:
        if os.path.exists(file_path):
            process_h5_file(file_path, device, args.batch_size)
        else:
            print(f"错误: 文件不存在 - {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='为H5文件中的k-mer序列添加ESM嵌入')
    parser.add_argument('--h5_files', nargs='+', 
                      default=[
                               '/xiongjun/test/NEW-MIL-dir-backup/backup/NEW-MIL-dir-backup/data/dataset/tcrensuredoinfo5_results_NewRank_C_h5_data/val.h5',
                               "/xiongjun/test/NEW-MIL-dir-backup/backup/NEW-MIL-dir-backup/data/dataset/tcrensuredoinfo5_results_NewRank_C_h5_data/test.h5",
                               "/xiongjun/test/NEW-MIL-dir-backup/backup/NEW-MIL-dir-backup/data/dataset/tcrensuredoinfo5_results_NewRank_C_h5_data/train.h5"],
                      help='要处理的H5文件路径')
    parser.add_argument('--batch_size', type=int, default=1000, help='处理序列的批量大小')
    
    args = parser.parse_args()
    main(args)