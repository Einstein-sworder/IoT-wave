# TFZ-Tree: An Ultra-Lightweight Waveform Classification Framework for Resource-Constrained Devices

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Datasets](https://img.shields.io/badge/Datasets-Available-green.svg)]

## 📥 Dataset Download
**Dataset File: IOT-wave_data**
Link: https://pan.baidu.com/s/1pHe5EbLR7q_AYyUCQqojNA
Password: `ez8e`

## Overview
This repository contains the datasets and supporting materials for the paper:

**TFZ-Tree: An Ultra-Lightweight Waveform Classification Framework for Resource-Constrained Devices**

The framework combines joint time–frequency features with a lightweight statistical decision tree, achieving highly efficient waveform recognition. It is designed for IoT and edge devices with strict computation and memory limits.

## Repository Contents
- **AWGN Channel Dataset**
- **TDL‑C Channel Dataset (3GPP multi‑path fading)**

The algorithm source code will be released upon paper acceptance.

## Dataset Description
| Channel | Description | Folder |
|---|---|---|
| **AWGN** | Additive White Gaussian Noise (clean baseline) | `data/awgn` |
| **TDL‑C** | 3GPP multi‑path fading | `data/tdlc` |

> Each dataset contains labeled I/Q samples for multiple modulation types. Detailed parameters are in `data/README.md`.

## Usage
```bash
git clone https://github.com/Einstein-sworder/IoT-wave.git
cd IoT-wave