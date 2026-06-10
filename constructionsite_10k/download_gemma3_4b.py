"""
下载 Gemma 3 4B Instruct（视觉语言模型）
"""
from modelscope import snapshot_download

BASE_DIR = "/root/autodl-tmp"  # 改这一行

print("📥 下载 Gemma-3-4B-IT (~9GB)...")
snapshot_download(
    'LLM-Research/gemma-3-4b-it',
    local_dir=f'{BASE_DIR}/gemma-3-4b-it'
)
print("✅ Gemma-3-4B-IT 下载完成！")
