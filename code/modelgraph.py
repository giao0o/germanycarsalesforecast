import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# 读取汇总数据
df = pd.read_excel(
    '/Users/jianaoli/germanycarsalesforecast/processed_reports/MG销售总报告.xlsx',
    engine='openpyxl',  # 必需参数
    parse_dates=['月份']
)
df['日期'] = pd.to_datetime(df['月份'] + '-01')  # 统一为每月第一天

def create_tech_indicators(ts_df):
    """为时间序列添加技术指标"""
    # 计算常见指标
    ts_df = ts_df.sort_index()
    
    # MACD
    macd = ta.trend.MACD(ts_df['总销量'], window_slow=26, window_fast=12, window_sign=9)
    ts_df['MACD'] = macd.macd()
    ts_df['Signal'] = macd.macd_signal()
    ts_df['Hist'] = macd.macd_diff()
    
    # 随机指标(KDJ)
    stochastic = ta.momentum.StochasticOscillator(
        high=ts_df['总销量'], 
        low=ts_df['总销量'],
        close=ts_df['总销量'],
        window=14,
        smooth_window=3
    )
    ts_df['K'] = stochastic.stoch()
    ts_df['D'] = stochastic.stoch_signal()
    
    # 威廉指标(WR)
    ts_df['WR'] = ta.momentum.WilliamsR(
        high=ts_df['总销量'], 
        low=ts_df['总销量'],
        close=ts_df['总销量'],
        lbp=14
    ).williams_r()
    
    # 移动平均线
    ts_df['MA3'] = ts_df['总销量'].rolling(3).mean()
    ts_df['MA5'] = ts_df['总销量'].rolling(5).mean()
    ts_df['MA12'] = ts_df['总销量'].rolling(12).mean()
    
    return ts_df

def plot_model_analysis(model_name, model_df):
    """生成单个车型的技术分析图"""
    model_df = model_df.set_index('日期').asfreq('MS').sort_index()
    tech_df = create_tech_indicators(model_df)
    
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
        go.Scatter(x=tech_df.index, y=tech_df['总销量'], name='销量'),
        row=1, col=1
    )
    for ma in ['MA3', 'MA5', 'MA12']:
        fig.add_trace(
            go.Scatter(x=tech_df.index, y=tech_df[ma], name=ma, line=dict(width=1)),
            row=1, col=1
        )
        
    # MACD
    fig.add_trace(
        go.Bar(x=tech_df.index, y=tech_df['Hist'], name='MACD柱', marker_color='grey'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=tech_df.index, y=tech_df['MACD'], name='MACD', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=tech_df.index, y=tech_df['Signal'], name='信号线', line=dict(color='orange')),
        row=2, col=1
    )
    
    # 随机指标KDJ
    fig.add_trace(
        go.Scatter(x=tech_df.index, y=tech_df['K'], name='K线', line=dict(color='green')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=tech_df.index, y=tech_df['D'], name='D线', line=dict(color='red')),
        row=3, col=1
    )
    
    # 威廉指标
    fig.add_trace(
        go.Scatter(x=tech_df.index, y=tech_df['WR'], name='WR', line=dict(color='purple')),
        row=4, col=1
    )
    
    fig.update_layout(
        title=f'{model_name} 销量技术分析',
        height=1000,
        showlegend=True,
        template='plotly_dark',
        hovermode='x unified'
    )
    
    # 保存为HTML
    output_dir = Path('技术分析图表')
    output_dir.mkdir(exist_ok=True)
    fig.write_html(str(output_dir/f'{model_name}_技术分析.html'))
    
    return fig

# 生成各车型分析报告
for model in df['车型'].unique():
    model_df = df[df['车型'] == model][['日期', '总销量']]
    plot_model_analysis(model, model_df)

# 生成总体销量分析
total_df = df.groupby('日期')['总销量'].sum().reset_index()
plot_model_analysis('总体销量', total_df)