#!/usr/bin/env python3
import os
from collections import Counter
from pathlib import Path

def count_extensions(root_dir):
    """
    递归统计 root_dir 下所有文件的扩展名。
    返回两个 Counter：full_ext（完整扩展名）和 simple_ext（简单扩展名）。
    """
    full_ext_counter = Counter()
    simple_ext_counter = Counter()
    
    root_path = Path(root_dir).resolve()
    if not root_path.is_dir():
        print(f"错误：{root_dir} 不是一个有效的目录")
        return full_ext_counter, simple_ext_counter
    
    # 递归遍历所有文件
    for file_path in root_path.rglob('*'):
        if file_path.is_file():
            name = file_path.name
            # 完整扩展名：从第一个点开始（如果有）
            if '.' in name:
                full_ext = '.' + name.split('.', 1)[1]   # 包含点
            else:
                full_ext = ''   # 无扩展名
            
            # 简单扩展名：最后一个点之后
            simple_ext = file_path.suffix   # 包含点，如 '.gz'
            
            full_ext_counter[full_ext] += 1
            simple_ext_counter[simple_ext] += 1
    
    return full_ext_counter, simple_ext_counter

if __name__ == "__main__":
    target_dir = "/data1/dataset/jssnu005/sys/MNBVC/"
    print(f"正在扫描目录: {target_dir}\n")
    
    full_cnt, simple_cnt = count_extensions(target_dir)
    
    print("=== 完整扩展名统计（从第一个点开始） ===")
    for ext, count in sorted(full_cnt.items(), key=lambda x: -x[1]):
        if ext == '':
            print(f"(无扩展名): {count}")
        else:
            print(f"{ext}: {count}")
    
    print("\n=== 简单扩展名统计（最后一个点之后） ===")
    for ext, count in sorted(simple_cnt.items(), key=lambda x: -x[1]):
        if ext == '':
            print(f"(无扩展名): {count}")
        else:
            print(f"{ext}: {count}")
    
    print(f"\n总文件数: {sum(full_cnt.values())}")