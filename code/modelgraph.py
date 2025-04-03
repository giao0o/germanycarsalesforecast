import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns  # 新增的导入
from pathlib import Path

# 配置参数
input_csv = "/Users/jianaoli/germanycarsalesforecast/processed_reports/MG销售总报告.csv"
output_dir = Path("sales_analysis")
output_dir.mkdir(exist_ok=True)

# 数据预处理
def preprocess_data(df):
    # 转换日期格式
    df['日期'] = pd.to_datetime(df['年份'].astype(str) + '-' + df['月份'].str.split('-').str[1])
    
    # 清洗车型名称
    df['车型'] = df['车型'].str.upper().str.strip()
    df['车型'] = df['车型'].replace({
        'SONSTIGE': '其他',
        '3': 'MG 3',
        '4': 'MG 4',
        '5': 'MG 5'
    })
    
    # 添加季度信息
    df['季度'] = df['日期'].dt.to_period('Q').astype(str)
    return df.sort_values('日期')

# 可视化分析
def generate_analysis(df):
    # 1. 总销量趋势（交互式）
    fig_total = px.area(
        df.groupby('日期')['总销量'].sum().reset_index().sort_values('日期'),
        x='日期',
        y='总销量',
        title='每月总销量趋势',
        labels={'总销量': '销量（辆）'},
        template='plotly_white'
    )
    fig_total.write_html(output_dir/"total_sales_trend.html")
    
    # 2. 车型销量排行榜（动态）
    top_models = df.groupby('车型')['总销量'].sum().nlargest(10).index
    ranking_df = df[df['车型'].isin(top_models)].groupby(['车型', '季度'])['总销量'].sum().reset_index()
    ranking_df['季度'] = pd.Categorical(ranking_df['季度'], categories=sorted(df['季度'].unique()), ordered=True)
    
    fig_ranking = px.bar(
        ranking_df.sort_values('季度'),
        x='季度',
        y='总销量',
        color='车型',
        barmode='stack',
        title='季度车型销量排行榜',
        labels={'总销量': '销量（辆）'},
        height=600
    )
    fig_ranking.write_html(output_dir/"model_ranking.html")

    # 3. 各车型销售趋势（分面图）
    trend_df = df.groupby(['日期', '车型'])['总销量'].sum().reset_index()
    trend_df = trend_df.sort_values('日期')
    
    fig_models = px.line(
        trend_df,
        x='日期',
        y='总销量',
        color='车型',
        facet_col='车型',
        facet_col_wrap=3,
        height=1500,
        title='各车型销量趋势'
    )
    fig_models.update_xaxes(matches=None, showticklabels=True)
    fig_models.write_html(output_dir/"model_trends.html")

    # 4. 热力图分析（静态）
    plt.figure(figsize=(16, 10))
    heatmap_data = df.pivot_table(
        index=df['日期'].dt.year,
        columns=df['日期'].dt.month,
        values='总销量',
        aggfunc='sum',
        fill_value=0
    ).astype(int)
    heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)  # 确保月份排序
    
    sns.heatmap(
        heatmap_data, 
        cmap='YlGnBu', 
        annot=True, 
        fmt="d",
        annot_kws={"size": 8},
        linewidths=.5
    )
    
    plt.title('Heatmap Sales Year-Month')
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.savefig(output_dir/"heatmap.png", dpi=300, bbox_inches='tight')
    
    print("分析报告已生成。")

if __name__ == "__main__":
    # 读取数据
    df = pd.read_csv(input_csv)
    
    # 数据预处理
    df = preprocess_data(df)
    
    # 生成分析报告
    generate_analysis(df)
    
    print(f"分析报告已生成至：{output_dir.absolute()}")
