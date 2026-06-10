"""
下载 InternVL2-4B 和 LLaVA-1.5-7B 到本地
运行：python download_internvl_llava.py
"""

from modelscope import snapshot_download

BASE_DIR = "/root/autodl-tmp"  # 改这一行

# ==================== 下载 InternVL2-4B ====================
print("📥 开始下载 InternVL2-4B (~8GB)...")
snapshot_download(
    'OpenGVLab/InternVL2-4B',
    local_dir=f'{BASE_DIR}/InternVL2-4B'
)
print("✅ InternVL2-4B 下载完成！\n")

# ==================== 下载 LLaVA-1.5-7B ====================
print("📥 开始下载 LLaVA-1.5-7B (~14GB)...")
snapshot_download(
    'AI-ModelScope/llava-v1.5-7b',
    local_dir=f'{BASE_DIR}/llava-v1.5-7b'
)
print("✅ LLaVA-1.5-7B 下载完成！")
