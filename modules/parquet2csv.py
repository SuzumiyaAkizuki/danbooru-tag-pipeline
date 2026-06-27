import pandas as pd


def parquet_to_csv(parquet_path, csv_path):
    """
    将 Parquet 文件转换为 CSV 文件

    参数:
    parquet_path (str): 输入的 Parquet 文件路径
    csv_path (str): 输出的 CSV 文件路径
    """
    try:
        # 1. 读取 Parquet 文件
        print(f"正在读取 Parquet 文件: {parquet_path}...")
        df = pd.read_parquet(parquet_path, engine='pyarrow')

        # 2. 保存为 CSV 文件
        # index=False 表示在 CSV 中不保存 DataFrame 的行索引
        print(f"正在保存为 CSV 文件: {csv_path}...")
        df.to_csv(csv_path, index=False, encoding='utf-8')

        print("转换成功！")

    except FileNotFoundError:
        print(f"错误：找不到文件 {parquet_path}，请检查路径是否正确。")
    except Exception as e:
        print(f"转换过程中出现错误: {e}")


def csv_to_parquet(csv_path, parquet_path):
    """
    将 CSV 文件转换为 Parquet 文件

    参数:
    csv_path (str): 输入的 CSV 文件路径
    parquet_path (str): 输出的 Parquet 文件路径
    """
    try:
        # 1. 读取 CSV 文件
        print(f"正在读取 CSV 文件: {csv_path}...")
        # low_memory=False 可以避免读取大文件时 pandas 产生的列数据类型推断警告
        df = pd.read_csv(csv_path, low_memory=False)

        # 2. 保存为 Parquet 文件
        print(f"正在保存为 Parquet 文件: {parquet_path}...")
        # compression='snappy' 是 Parquet 默认且推荐的压缩算法，兼顾了压缩率和读写速度
        df.to_parquet(parquet_path, engine='pyarrow', index=False, compression='snappy')

        print("转换成功！")

    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_path}，请检查路径是否正确。")
    except UnicodeDecodeError:
        print("错误：文件编码可能不是 UTF-8。可以尝试在 read_csv 中指定 encoding='gbk' 等参数。")
    except Exception as e:
        print(f"转换过程中出现错误: {e}")

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 将这里替换为你实际的文件路径
    csv_path = "wiki_pages.csv"
    parquet_path = "wiki_pages.parquet"

    parquet_to_csv(parquet_path,csv_path)