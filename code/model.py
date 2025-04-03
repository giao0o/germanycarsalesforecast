import pandas as pd
import os
import re
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# 本程序用于从KBA下载的德国各品牌汽车每月销售统计表（FZ10）自动整理相应的数据为csv和xlsx格式，以做进一步分析，如MG品牌所有车型
# 配置路径
input_dir = "./kba_downloads"
output_dir = "./processed_reports"
os.makedirs(output_dir, exist_ok=True)

def find_sheet_name(file_path):
    """动态查找包含FZ 10.1的工作表（支持多种格式）"""
    try:
        with pd.ExcelFile(file_path) as xls:
            # 匹配多种可能的格式：FZ 10.1 / FZ10.1 / FZ_10.1等
            pattern = re.compile(r'FZ[ _-]*10[\. _-]*1', flags=re.IGNORECASE)
            for sheet in xls.sheet_names:
                if pattern.search(sheet):
                    return sheet
            return None
    except Exception as e:
        print(f"文件读取失败：{os.path.basename(file_path)} - {str(e)}")
        return None

def process_single_file(file_path):
    try:
        sheet_name = find_sheet_name(file_path)
        if not sheet_name:
            return pd.DataFrame()
        
        # 读取数据（调整skiprows值）
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            usecols="B:D",
            header=None,
            skiprows=5,  # 可能需要调整为4或6，根据实际表头位置
            engine='openpyxl'
        )
        df.columns = ['品牌', '车型', '销量']
        
        # 关键修复步骤：填充品牌列
        df['品牌'] = df['品牌'].fillna(method='ffill')
        
        # 筛选目标品牌
        mask = df['品牌'].str.contains(
            r'^MG\s*ROEWE$', 
            case=False, 
            na=False, 
            regex=True
        )
        target_df = df[mask].copy()
        
        if target_df.empty:
            return pd.DataFrame()
        
        # 数据清洗
        target_df['车型'] = (
            target_df['车型']
            .str.strip()
            .str.upper()
            .str.replace(r'\s+', ' ', regex=True)
            .str.replace(r'^(\d+)([A-Z]+)$', r'\1 \2', regex=True)  # 分离数字和字母
        )
        
        # 销量处理
        target_df['销量'] = (
            pd.to_numeric(target_df['销量'], errors='coerce')
            .fillna(0)
            .astype(int)
        )
        
        # 合并相同车型
        result = (
            target_df.groupby('车型', as_index=False)
            .agg(总销量=('销量', 'sum'))
            .sort_values('总销量', ascending=False)
        )
        
        return result
    
    except Exception as e:
        print(f"处理失败：{os.path.basename(file_path)} - {str(e)}")
        return pd.DataFrame()

def batch_processing():
    all_files = glob(os.path.join(input_dir, "*.xlsx"))
    final_data = []
    
    for file in all_files:
        filename = os.path.basename(file)
        print(f"\n正在处理：{filename}")
        
        # 提取年份月份
        match = re.search(r'_(\d{4})_(\d{2})\.xlsx$', filename)
        if not match:
            print(f"跳过无效文件名格式：{filename}")
            continue
            
        year, month = match.groups()
        month_str = f"{year}-{month}"
        
        # 处理文件
        df = process_single_file(file)
        if df.empty:
            print("未找到有效数据")
            continue
        
        # 添加时间维度
        df.insert(0, '年份', year)
        df.insert(1, '月份', month_str)
        
        # 保存月度报告
        report_name = f"MG_{year}_{month}_report.csv"
        df.to_csv(os.path.join(output_dir, report_name), index=False)
        
        final_data.append(df)
        print(f"找到 {len(df)} 条车型记录")
    
    if final_data:
        full_report = pd.concat(final_data)
        
        # 生成总CSV报告
        full_csv_path = os.path.join(output_dir, 'MG销售总报告.csv')
        full_report.sort_values(by=['年份', '月份']).to_csv(full_csv_path, index=False)
        print(f"总CSV报告已保存至：{full_csv_path}")

        # 生成各车型独立报告
        model_dir = os.path.join(output_dir, '车型报告')
        os.makedirs(model_dir, exist_ok=True)
        
        # 按车型和时间排序
        sorted_report = full_report.sort_values(by=['车型', '年份', '月份'])
        
        for model in sorted_report['车型'].unique():
            # 清洗车型名称用于文件名
            clean_name = re.sub(r'[\\/*?:"<>|]', '', model).strip()[:50]
            
            # 按时间排序并保存
            model_df = sorted_report[sorted_report['车型'] == model] \
                .sort_values(by=['年份', '月份']) \
                [['年份', '月份', '总销量']]
            
            csv_path = os.path.join(model_dir, f'MG_{clean_name}_销量报告.csv')
            model_df.to_csv(csv_path, index=False)
        
        print(f"各车型报告已保存至：{model_dir}")
        # 生成总报告
        
        
        # 多维分析
        with pd.ExcelWriter(os.path.join(output_dir, 'MG销售总报告.xlsx')) as writer:
            # 原始数据
            full_report.to_excel(writer, sheet_name='原始数据', index=False)
            
            # 时间趋势透视表
            pivot_table = pd.pivot_table(
                full_report,
                values='总销量',
                index='车型',
                columns='月份',
                aggfunc='sum',
                fill_value=0
            )
            pivot_table.to_excel(writer, sheet_name='月度趋势')
            
            # 车型总排名
            (full_report.groupby('车型')['总销量'].sum()
             .sort_values(ascending=False)
             .to_excel(writer, sheet_name='车型排名'))
            
            # 年度汇总
            (full_report.groupby(['年份', '车型'])['总销量'].sum()
             .unstack().fillna(0)
             .to_excel(writer, sheet_name='年度汇总'))
            
        print(f"\n处理完成！共处理 {len(final_data)} 个有效文件")
        print(f"总报告已保存至：{os.path.abspath(output_dir)}/MG销售总报告.xlsx")
    else:
        print("\n未找到任何有效数据")

if __name__ == "__main__":
    batch_processing()