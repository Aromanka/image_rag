from modelscope import snapshot_download

print("开始下载 Qwen3-VL-4B-Instruct...")

snapshot_download(
    'qwen/Qwen3-VL-4B-Instruct',
    local_dir = "/root/autodl-tmp"  # 改这一行
)

print("✅ 下载完成！")
