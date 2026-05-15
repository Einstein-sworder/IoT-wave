"""
读取 waveform_modrec_tdlc_16snr_4speed_2ds_6mod_10wave_mfsk_balanced.npz
并打印类型、形状等说明信息。

数据生成脚本: LDT-C/data.ipynb (Sionna TDL-C多径信道, TensorFlow GPU)
生成参数:
  - 波形类型: 10类 (OFDM/OTFS/ODDM/FBMC/UFMC/DSSS/LoRa/NB-IoT/GFSK/MFSK)
  - SNR档数: 16档 (0,2,4,...,30 dB)
  - 速度档数: 4档 (30, 90, 150, 210 km/h)
  - 时延扩展: 2档 (300ns, 600ns)
  - 调制类型: 6类 (QPSK/8PSK/32QAM/64APSK/256QAM/4096QAM)
  - 每组合基础样本数: 20 (GFSK/MFSK重复3次以平衡总数)
  - 信号长度: 1024 复数点
  - 信道: TDL-C多径衰落 + AWGN
  - 总样本数: 16×480×20 = 153600 (每SNR 9600)
"""

import numpy as np

NPZ_PATH = "LDT-C/waveform_modrec_tdlc_16snr_4speed_2ds_6mod_10wave_mfsk_balanced.npz"

print("=" * 72)
print(f"读取文件: {NPZ_PATH}")
print("=" * 72)

data = np.load(NPZ_PATH, allow_pickle=True, mmap_mode='r')

print(f"\n文件内包含的键 (keys): {list(data.keys())}")

# ---------- 获取各数组 ----------
raw = data["data"]
wave_labels = data.get("wave_labels", None)
speed_labels = data.get("speed_labels", None)
delay_spread_labels = data.get("delay_spread_labels", None)
mod_labels = data.get("mod_labels", None)
snr_list = data.get("snr_list", None)
velocity_list = data.get("velocity_list", None)
delay_spread_list = data.get("delay_spread_list", None)
modulation_names = data.get("modulation_names", None)
waveform_names = data.get("waveform_names", None)
samples_per_combo = data.get("samples_per_combo", None)
signal_length = data.get("signal_length", None)

# ---------- 打印各字段 ----------
print(f"\n--- data 数组 (接收信号) ---")
print(f"  dtype:    {raw.dtype}")
print(f"  shape:    {raw.shape}")
print(f"  内存大小: {raw.nbytes / 1024**3:.2f} GB")
num_snr, total_per_snr, sig_len = raw.shape
print(f"  含义: (SNR档数={num_snr}, 每SNR样本数={total_per_snr}, 信号长度={sig_len})")

print(f"\n--- wave_labels (波形标签) ---")
if wave_labels is not None:
    print(f"  dtype:    {wave_labels.dtype}")
    print(f"  shape:    {wave_labels.shape}")
    unique, cnts = np.unique(wave_labels, return_counts=True)
    print(f"  唯一标签数: {len(unique)}")
    for u, c in zip(unique, cnts):
        wname = waveform_names[u] if waveform_names is not None else f"wave_{u}"
        print(f"    label {u} ({wname}): {c}")

print(f"\n--- mod_labels (调制标签) ---")
if mod_labels is not None:
    print(f"  dtype:    {mod_labels.dtype}")
    print(f"  shape:    {mod_labels.shape}")
    unique_m, cnts_m = np.unique(mod_labels, return_counts=True)
    print(f"  唯一标签数: {len(unique_m)}")
    for u, c in zip(unique_m, cnts_m):
        mname = modulation_names[u] if modulation_names is not None else f"mod_{u}"
        print(f"    label {u} ({mname}): {c}")

print(f"\n--- speed_labels (速度标签) ---")
if speed_labels is not None:
    print(f"  dtype:    {speed_labels.dtype}")
    print(f"  shape:    {speed_labels.shape}")
    unique_s, cnts_s = np.unique(speed_labels, return_counts=True)
    print(f"  唯一标签数: {len(unique_s)}")
    for u, c in zip(unique_s, cnts_s):
        vname = f"{velocity_list[u]} km/h" if velocity_list is not None else f"speed_{u}"
        print(f"    label {u} ({vname}): {c}")

print(f"\n--- delay_spread_labels (时延扩展标签) ---")
if delay_spread_labels is not None:
    print(f"  dtype:    {delay_spread_labels.dtype}")
    print(f"  shape:    {delay_spread_labels.shape}")
    unique_d, cnts_d = np.unique(delay_spread_labels, return_counts=True)
    print(f"  唯一标签数: {len(unique_d)}")
    for u, c in zip(unique_d, cnts_d):
        dname = f"{delay_spread_list[u]*1e9:.0f} ns" if delay_spread_list is not None else f"ds_{u}"
        print(f"    label {u} ({dname}): {c}")

print(f"\n--- snr_list ---")
if snr_list is not None:
    print(f"  dtype: {snr_list.dtype}, shape: {snr_list.shape}")
    print(f"  内容: {snr_list.tolist()}")

print(f"\n--- velocity_list ---")
if velocity_list is not None:
    print(f"  dtype: {velocity_list.dtype}, shape: {velocity_list.shape}")
    print(f"  内容: {velocity_list.tolist()} km/h")

print(f"\n--- delay_spread_list ---")
if delay_spread_list is not None:
    print(f"  dtype: {delay_spread_list.dtype}, shape: {delay_spread_list.shape}")
    print(f"  内容: {(delay_spread_list * 1e9).tolist()} ns")

print(f"\n--- modulation_names ---")
if modulation_names is not None:
    print(f"  dtype: {modulation_names.dtype}, shape: {modulation_names.shape}")
    print(f"  内容: {[str(x) for x in modulation_names.tolist()]}")

print(f"\n--- waveform_names ---")
if waveform_names is not None:
    print(f"  dtype: {waveform_names.dtype}, shape: {waveform_names.shape}")
    print(f"  内容: {[str(x) for x in waveform_names.tolist()]}")

print(f"\n--- samples_per_combo ---")
if samples_per_combo is not None:
    print(f"  值: {samples_per_combo}")

print(f"\n--- signal_length ---")
if signal_length is not None:
    print(f"  值: {signal_length}")

# ---------- 逐维度统计 (使用切片避免全量加载) ----------
print(f"\n--- 数据统计 (基于抽样) ---")
total_samples = num_snr * total_per_snr
print(f"  总样本数: {total_samples}")
# 仅统计第1个SNR层级下的前1000个样本，避免内存溢出
sample_slice = raw[0, :1000, :]
print(f"  信号实部均值 (抽样): {sample_slice.real.mean():.6f}")
print(f"  信号实部标准差 (抽样): {sample_slice.real.std():.6f}")
print(f"  信号虚部均值 (抽样): {sample_slice.imag.mean():.6f}")
print(f"  信号虚部标准差 (抽样): {sample_slice.imag.std():.6f}")
print(f"  信号平均功率 (抽样): {np.mean(np.abs(sample_slice)**2):.6f}")

# ---------- 按波形×调制交叉统计 ----------
if wave_labels is not None and mod_labels is not None:
    wl = wave_labels.flatten()
    ml = mod_labels.flatten()
    print(f"\n--- 波形×调制交叉分布 (每个组合的样本数) ---")
    n_wave = len(np.unique(wl))
    n_mod = len(np.unique(ml))
    print(f"  调制\\波形 | {' '.join(f'{waveform_names[w]:>6s}' if waveform_names is not None else f'w{w:>6}' for w in range(n_wave))}")
    print(f"  " + "-" * (12 + 7 * n_wave))
    for m in range(n_mod):
        mname = modulation_names[m] if modulation_names is not None else f"m{m}"
        row = f"  {mname:>10s} |"
        for w in range(n_wave):
            cnt = int(np.sum((wl == w) & (ml == m)))
            row += f" {cnt:>6d}"
        print(row)

# ---------- 数据样例 ----------
print(f"\n--- 数据样例 (前3个样本各取前5点) ---")
wl = wave_labels.flatten() if wave_labels is not None else None
ml = mod_labels.flatten() if mod_labels is not None else None
sl = speed_labels.flatten() if speed_labels is not None else None
dl = delay_spread_labels.flatten() if delay_spread_labels is not None else None
for i in range(min(3, total_samples)):
    snr_idx = i // total_per_snr
    offset = i % total_per_snr
    sample_i = raw[snr_idx, offset, :5]
    wi = wl[i] if wl is not None else -1
    mi = ml[i] if ml is not None else -1
    si = sl[i] if sl is not None else -1
    di = dl[i] if dl is not None else -1
    wname = waveform_names[wi] if waveform_names is not None else str(wi)
    mname = modulation_names[mi] if modulation_names is not None else str(mi)
    vname = f"{velocity_list[si]}km/h" if velocity_list is not None else str(si)
    dname = f"{delay_spread_list[di]*1e9:.0f}ns" if delay_spread_list is not None else str(di)
    print(f"  sample[{i}]: 波形={wname}, 调制={mname}, 速度={vname}, 时延={dname}")
    print(f"    {sample_i}")

print("\n" + "=" * 72)
print("数据集摘要:")
print(f"  波形类型数:   10 (OFDM/OTFS/ODDM/FBMC/UFMC/DSSS/LoRa/NB-IoT/GFSK/MFSK)")
print(f"  调制类型数:   6 (QPSK/8PSK/32QAM/64APSK/256QAM/4096QAM)")
print(f"  SNR档数:     {num_snr} (0~30 dB, 步长2)")
print(f"  速度档数:    4 (30/90/150/210 km/h)")
print(f"  时延扩展档数: 2 (300ns/600ns)")
print(f"  每组合样本:  {samples_per_combo} (GFSK/MFSK重复3倍以平衡)")
print(f"  每SNR样本数: {total_per_snr} (均衡后)")
print(f"  总样本数:    {total_samples}")
print(f"  信号长度:    {sig_len} 复数点")
print(f"  信道类型:    TDL-C 多径衰落 + AWGN")
print(f"  生成框架:    Sionna (TensorFlow) GPU")
print("=" * 72)
