import os
import csv
import re
import glob

# Resolve project root relative to this script's location
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results_advanced")
PBR_DIR = os.path.join(RESULTS_DIR, "pbr")
DEBACKER_DIR = os.path.join(RESULTS_DIR, "debacker")
VIDEOS_DIR = os.path.join(RESULTS_DIR, "videos")
RESULTS_TXT = os.path.join(RESULTS_DIR, "结果.txt")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "all_results_combined.csv")

def parse_results_txt():
    """解析 结果.txt 文件，提取每条记录"""
    records = []
    with open(RESULTS_TXT, 'r', encoding='utf-8-sig', errors='replace') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        # 匹配格式: 视频名  PBR均值±PBR标准差  TVD  DeBacker  [OK]
        match = re.match(r'^(\S+(?:\s+\d+)?)\s+([\d.]+)±([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[OK\]', line)
        if match:
            video_name = match.group(1).strip()
            pbr_mean = float(match.group(2))
            pbr_std = float(match.group(3))
            tvd = float(match.group(4))
            debacker = float(match.group(5))
            records.append({
                'video_name': video_name,
                'pbr_mean': pbr_mean,
                'pbr_std': pbr_std,
                'tvd': tvd,
                'debacker': debacker,
                'status': 'OK'
            })
    
    return records

def read_speeds(video_name):
    """读取血流速度结果"""
    pattern = os.path.join(VIDEOS_DIR, "**", video_name, f"{video_name}_speeds.csv")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None, None
    speed_file = matches[0]
    with open(speed_file, 'r', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.reader(f)
        rows = list(reader)
    mean_speed = std_speed = None
    for row in rows:
        if len(row) >= 2:
            label = row[0].strip()
            try:
                val = float(row[1])
                if "Overall Mean Speed" in label:
                    mean_speed = val
                elif "Overall Speed Std" in label:
                    std_speed = val
            except ValueError:
                pass
    return mean_speed, std_speed

def main():
    records = parse_results_txt()
    print(f"从 结果.txt 解析到 {len(records)} 条记录")
    
    # 写入 CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Video Name",
            "PBR Mean (μm)", "PBR Std (μm)",
            "TVD",
            "De Backer Score",
            "Speed Mean (μm/s)", "Speed Std (μm/s)",
            "Status"
        ])
        
        for rec in records:
            speed_mean, speed_std = read_speeds(rec['video_name'])
            
            writer.writerow([
                rec['video_name'],
                f"{rec['pbr_mean']:.2f}",
                f"{rec['pbr_std']:.2f}",
                f"{rec['tvd']:.2f}",
                f"{rec['debacker']:.2f}",
                f"{speed_mean:.2f}" if speed_mean is not None else "N/A",
                f"{speed_std:.2f}" if speed_std is not None else "N/A",
                rec['status']
            ])
    
    print(f"Combined results saved to: {OUTPUT_CSV}")
    print(f"Total records: {len(records)}")

if __name__ == "__main__":
    main()
