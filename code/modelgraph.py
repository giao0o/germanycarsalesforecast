import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ======================
# 数据读取与预处理
# ======================
def load_data():
    """读取总报告数据"""
    file_path = Path("/Users/jianaoli/germanycarsalesforecast/processed_reports/MG销售总报告.xlsx")
    
    # 读取时保持月份为原始字符串格式
    df = pd.read_excel(
        file_path,
        engine='openpyxl',
        dtype={'月份': str}  # 关键修复：强制保留原始字符串格式
    )
    
    # 转换为日期（处理"2025-02" -> "2025-02-01"）
    df['日期'] = pd.to_datetime(
        df['月份'] + '-01',  # 拼接日期部分
        format='%Y-%m-%d',
        errors='coerce'
    )
    
    # 验证日期转换
    if df['日期'].isnull().any():
        invalid_data = df[df['日期'].isnull()]
        print("发现无效月份格式：")
        print(invalid_data[['年份', '月份', '车型', '总销量']])
        df = df.dropna(subset=['日期'])
    
    return df

# ======================
# 技术指标计算
# ======================
def calculate_technical_indicators(df):
    """计算技术分析指标"""
    # 准备时间序列数据
    ts_df = df.set_index('日期').sort_index()
    
    # MACD指标
    macd = ta.trend.MACD(ts_df['总销量'], window_slow=26, window_fast=12, window_sign=9)
    ts_df['MACD'] = macd.macd()
    ts_df['MACD_Signal'] = macd.macd_signal()
    ts_df['MACD_Hist'] = macd.macd_diff()
    
    # 随机指标KDJ
    stochastic = ta.momentum.StochasticOscillator(
        high=ts_df['总销量'], 
        low=ts_df['总销量'],
        close=ts_df['总销量'],
        window=14,
        smooth_window=3
    )
    ts_df['K'] = stochastic.stoch()
    ts_df['D'] = stochastic.stoch_signal()
    
    # # 威廉指标
    # ts_df['WR'] = ta.momentum.WilliamsR(
    #     high=ts_df['总销量'], 
    #     low=ts_df['总销量'],
    #     close=ts_df['总销量'],
    #     lbp=14
    # ).williams_r()
    
    # 移动平均线
    for period in [3, 5, 12]:
        ts_df[f'MA{period}'] = ts_df['总销量'].rolling(window=period).mean()
    
    return ts_df.reset_index()

# ======================
# 可视化模块
# ======================
def generate_technical_chart(model_name, df):
    """生成技术分析图表"""
    # 创建多子图布局
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.2, 0.2],
        specs=[[{"secondary_y": True}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # 主图：销量和均线
    fig.add_trace(
        go.Scatter(x=df['日期'], y=df['总销量'], name='实际销量', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['日期'], y=df['MA3'], name='3月均线', line=dict(color='#ff7f0e', dash='dot')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['日期'], y=df['MA5'], name='5月均线', line=dict(color='#2ca02c', dash='dash')),
        row=1, col=1
    )
    
    # MACD指标
    colors = ['green' if val > 0 else 'red' for val in df['MACD_Hist']]
    fig.add_trace(
        go.Bar(x=df['日期'], y=df['MACD_Hist'], name='MACD柱', marker_color=colors),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['日期'], y=df['MACD'], name='MACD', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['日期'], y=df['MACD_Signal'], name='信号线', line=dict(color='orange')),
        row=2, col=1
    )
    
    # KDJ指标
    fig.add_trace(
        go.Scatter(x=df['日期'], y=df['K'], name='K线', line=dict(color='#d62728')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['日期'], y=df['D'], name='D线', line=dict(color='#9467bd')),
        row=3, col=1
    )
    
    # # 威廉指标
    # fig.add_trace(
    #     go.Scatter(x=df['日期'], y=df['WR'], name='WR', line=dict(color='#8c564b')),
    #     row=4, col=1
    # )
    
    # 图表样式设置
    fig.update_layout(
        title=f'{model_name} 销量技术分析',
        height=1000,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # 坐标轴标签
    fig.update_yaxes(title_text="销量", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="KDJ", row=3, col=1)
    fig.update_yaxes(title_text="WR", row=4, col=1)
    fig.update_xaxes(title_text="日期", row=4, col=1)
    
    # 保存图表
    output_dir = Path('技术分析图表')
    output_dir.mkdir(exist_ok=True)
    fig.write_html(str(output_dir/f'{model_name}_技术分析.html'))
    fig.write_image(str(output_dir/f'{model_name}_技术分析.png'), scale=2)
    
    return fig

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    # 加载数据
    df = load_data()
    
    # 生成各车型报告
    for model in df['车型'].unique():
        print(f"\n正在生成 {model} 的分析报告...")
        model_df = df[df['车型'] == model]
        tech_df = calculate_technical_indicators(model_df)
        generate_technical_chart(model, tech_df)
    
    # 生成总体报告
    print("\n正在生成总体销量分析...")
    total_df = df.groupby('日期').agg({'总销量':'sum'}).reset_index()
    total_tech = calculate_technical_indicators(total_df)
    generate_technical_chart('总体销量', total_tech)
    
    print("\n所有分析报告已生成在 [技术分析图表] 目录")