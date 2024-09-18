import os
import requests
from tqdm import tqdm

# COCO数据集的下载链接，这里只是一个示例，需要替换为实际的下载链接
DOWNLOAD_URLS = [
    'http://images.cocodataset.org/zips/train2017.zip',
    'http://images.cocodataset.org/zips/val2017.zip',
    # ... 其他需要下载的文件链接
]

# 下载文件的保存路径
SAVE_DIR = './data'

# 重试次数
RETRY_TIMES = 3

# 断点续传的缓冲区大小
BUFFER_SIZE = 1024 * 1024  # 1MB

def download_file(url, save_path):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    local_filename = url.split('/')[-1]
    file_path = os.path.join(SAVE_DIR, local_filename)

    with requests.Session() as session:
        for i in range(RETRY_TIMES):
            try:
                response = session.get(url, stream=True)
                response.raise_for_status()

                # 检查文件是否已部分下载
                if os.path.exists(file_path):
                    initial_length = os.path.getsize(file_path)
                else:
                    initial_length = 0

                # 设置请求头，从中断的地方开始下载
                headers = {'Range': f'bytes={initial_length}-'}
                response = session.get(url, stream=True, headers=headers)
                response.raise_for_status()

                total_size = int(response.headers.get('Content-Length', 0))

                with open(file_path, 'ab') as f:
                    progress = tqdm(total=total_size, initial=initial_length, unit='B', unit_scale=True, desc=local_filename)
                    for chunk in response.iter_content(chunk_size=BUFFER_SIZE):
                        if chunk:  # filter out keep-alive new chunks
                            progress.update(len(chunk))
                            f.write(chunk)
                progress.close()

                if progress.n == total_size:
                    break

            except requests.RequestException as e:
                print(f'Download failed: {e}, retrying... ({i+1}/{RETRY_TIMES})')
                if i + 1 == RETRY_TIMES:
                    print('Max retries reached. Download failed.')
                    return False

    return True

def main():
    for url in DOWNLOAD_URLS:
        download_file(url, SAVE_DIR)

if __name__ == '__main__':
    main()