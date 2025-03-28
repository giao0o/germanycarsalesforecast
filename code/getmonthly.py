import requests
import os
import time

# 创建下载目录
download_dir = "./kba_downloads_monthly"
os.makedirs(download_dir, exist_ok=True)

# 设置请求头
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

def download_files():
    # 遍历年份和月份
    for year in range(2021, 2026):  # 2025+1=2026
        for month in range(1, 13):
            # 格式化月份为两位数
            month_str = f"{month:02d}"
            
            # 构造文件名和URL
            filename = f"fz10_{year}_{month_str}.xlsx"
            url = f"https://www.kba.de/SharedDocs/Downloads/DE/Statistik/Fahrzeuge/FZ10/{filename}?__blob=publicationFile&v=4"
            
            # 构造完整保存路径
            save_path = os.path.join(download_dir, filename)
            
            try:
                # 发送带流式传输的GET请求
                response = requests.get(url, headers=headers, stream=True, timeout=10)
                
                # 检查响应状态
                if response.status_code == 200:
                    # 写入文件
                    with open(save_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    print(f"成功下载：{filename}")
                else:
                    print(f"文件不存在：{filename}（状态码：{response.status_code}）")
                    
            except Exception as e:
                print(f"下载失败：{filename}，错误：{str(e)}")
            
            # 礼貌性等待
            time.sleep(1)

if __name__ == "__main__":
    download_files()
    print("下载任务完成！")