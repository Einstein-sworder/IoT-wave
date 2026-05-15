"""
读取 modulation_dataset_iot_10classes.npz 并打印类型、形状等说明信息。

数据生成脚本: data copy 2.ipynb (AWGN信道, CPU/CuPy)
生成参数:
  - 调制类型: 10类 (OFDM/OTFS/ODDM/FBMC/UFMC/DSSS/LoRa/NB-IoT/GFSK/MFSK)
  - SNR档数: 16档 (0,2,4,...,30 dB)
  - 调制阶数: 6阶 (QPSK/8PSK/32QAM/256QAM/64APSK/4096QAM)
  - 每组合样本数: 25
  - 信号长度: 1024 复数点
  - 信道: 纯AWGN, 无多径衰落
  - 总样本数: 10×16×6×25 = 24000
"""

import numpy as np

NPZ_PATH = "modulation_dataset_iot_10classes.npz"

print("=" * 72)
print(f"读取文件: {NPZ_PATH}")
print("=" * 72)

data = np.load(NPZ_PATH, allow_pickle=True)

print(f"\n文件内包含的键 (keys): {list(data.keys())}")

# ---------- 获取各数组 ----------
raw = data["data"]
labels = data["labels"]
mod_names = data.get("modulation_names", None)
snr_list = data.get("snr_list", None)
mod_orders = data.get("mod_orders", None)
shape_info = data.get("shape_info", None)

# ---------- 打印各字段 ----------
print(f"\n--- data 数组 ---")
print(f"  dtype:    {raw.dtype}")
print(f"  shape:    {raw.shape}")
print(f"  内存大小: {raw.nbytes / 1024**2:.2f} MB")
if raw.ndim == 2:
    print(f"  含义: (n_samples={raw.shape[0]}, signal_len={raw.shape[1]}) — 已展平为2D")
elif raw.ndim == 5:
    print(f"  含义: (mod_type={raw.shape[0]}, SNR={raw.shape[1]}, mod_order={raw.shape[2]}, samples={raw.shape[3]}, signal_len={raw.shape[4]})")

print(f"\n--- labels 数组 ---")
print(f"  dtype:    {labels.dtype}")
print(f"  shape:    {labels.shape}")
unique, counts = np.unique(labels, return_counts=True)
print(f"  唯一标签数: {len(unique)}")
for u, c in zip(unique, counts):
    name = mod_names[u] if mod_names is not None else f"class_{u}"
    print(f"    label {u} ({name}): {c} 样本")

print(f"\n--- modulation_names ---")
if mod_names is not None:
    print(f"  dtype: {mod_names.dtype}")
    print(f"  shape: {mod_names.shape}")
    print(f"  内容: {[str(x) for x in mod_names.tolist()]}")
else:
    print("  (不存在)")

print(f"\n--- snr_list ---")
if snr_list is not None:
    print(f"  dtype: {snr_list.dtype}, shape: {snr_list.shape}")
    print(f"  内容: {snr_list.tolist()}")
else:
    print("  (不存在)")

print(f"\n--- mod_orders ---")
if mod_orders is not None:
    print(f"  dtype: {mod_orders.dtype}, shape: {mod_orders.shape}")
    print(f"  内容: {[str(x) for x in mod_orders.tolist()]}")
else:
    print("  (不存在)")

print(f"\n--- shape_info ---")
if shape_info is not None:
    print(f"  dtype: {shape_info.dtype}, shape: {shape_info.shape}")
    print(f"  内容: {shape_info.tolist()}  (原始5D形状: mod×SNR×order×samp×len)")
else:
    print("  (不存在)")

# ---------- 数据样例 ----------
print(f"\n--- 数据样例 (前3个样本各取前5点) ---")
for i in range(min(3, raw.shape[0])):
    print(f"  sample[{i}] (label={labels[i]}, class={mod_names[labels[i]] if mod_names is not None else labels[i]}):")
    print(f"    {raw[i, :5]}")

print(f"\n--- 统计信息 ---")
print(f"  信号实部均值: {raw.real.mean():.6f}")
print(f"  信号实部标准差: {raw.real.std():.6f}")
print(f"  信号虚部均值: {raw.imag.mean():.6f}")
print(f"  信号虚部标准差: {raw.imag.std():.6f}")
print(f"  信号平均功率: {np.mean(np.abs(raw)**2):.6f}")

print("\n" + "=" * 72)
print("数据集摘要:")
print(f"  类别数:     10 (波形调制类型)")
print(f"  SNR档数:   16 (0~30 dB, 步长2)")
print(f"  调制阶数:   6 (QPSK/8PSK/32QAM/256QAM/64APSK/4096QAM)")
print(f"  每组合样本: 25")
print(f"  总样本数:   {raw.shape[0]}")
print(f"  信号长度:   {raw.shape[1]} 复数点")
print(f"  信道类型:   AWGN (纯加性高斯白噪声)")
print(f"  生成框架:   NumPy + CuPy (可选GPU)")
print("=" * 72)
