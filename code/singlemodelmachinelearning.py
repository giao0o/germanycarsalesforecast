import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 本程序通过使用机器学习模型预测未来3-5个月的某车系的月销量
# 配置参数
# 修改为对应csv文件
data_path = "/Users/jianaoli/germanycarsalesforecast/processed_reports/车型报告/MG_5_销量报告.csv"
# 修改为对应文件夹
output_dir = Path("sales_forecast_mg5")
output_dir.mkdir(exist_ok=True)

def create_features(df):
    """创建时序特征"""
    # 步骤1：验证基础列存在
    required_columns = ['年份', '月份', '总销量']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"数据中缺少必要列：{missing_cols}")

    # 步骤2：安全拆分月份
    try:
        month_parts = df['月份'].str.split('-', expand=True)
        df['年份_月份'] = month_parts[1]  # 获取第二部分如"11"
    except:
        raise ValueError("月份列格式异常，应为类似'2021-11'的格式")

    # 步骤3：生成日期列（带异常处理）
    try:
        df['month'] = pd.to_datetime(
            df['年份'].astype(str) + '-' + 
            df['年份_月份'] + '-01',
            format='%Y-%m-%d',
            errors='coerce'
        )
    except Exception as e:
        raise ValueError(f"日期生成失败：{str(e)}")

    # 检查无效日期
    if df['month'].isnull().any():
        invalid_data = df[df['month'].isnull()]
        print("发现无效日期数据：")
        print(invalid_data[['年份', '月份', '年份_月份']])
        df = df.dropna(subset=['month'])

    # 步骤4：设置时间索引
    df = df.set_index('month').sort_index().drop(columns=['年份_月份'])
    
    # 步骤5：创建时序特征
    try:
        # 滞后特征
        for lag in [1, 2, 3, 12]:
            df[f'lag_{lag}'] = df['总销量'].shift(lag)
        
        # 滚动特征
        df['rolling_3_mean'] = df['总销量'].shift(1).rolling(3).mean()
        df['rolling_6_std'] = df['总销量'].shift(1).rolling(6).std()
        
        # 时间特征
        df['month_num'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # 季节特征
        df['is_dec'] = (df.index.month == 12).astype(int)
    except Exception as e:
        raise ValueError(f"特征工程失败：{str(e)}")

    return df.dropna()
    

def train_model(X_train, y_train):
    """训练随机森林模型"""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        min_samples_split=3
    )
    model.fit(X_train, y_train)
    return model

def forecast(model, last_known_data, steps=1):
    """递归预测未来销量"""
    forecasts = []
    current_data = last_known_data.copy()
    
    for _ in range(steps):
        # 生成预测
        pred = model.predict(current_data)
        forecasts.append(pred[0])
        
        # 更新特征
        current_data['lag_1'] = pred[0]
        current_data['lag_2'] = current_data['lag_1']
        current_data['lag_3'] = current_data['lag_2']
        current_data['rolling_3_mean'] = np.mean([
            current_data['lag_1'],
            current_data['lag_2'],
            current_data['lag_3']
        ])
    return forecasts

def main():
    # 加载数据
    df = pd.read_csv(data_path)
    
    # 特征工程
    feature_df = create_features(df)
    
    # 准备数据
    X = feature_df.drop(columns=['总销量', '年份', '月份'])
    y = feature_df['总销量']
    
    # 时间序列分割
    tscv = TimeSeriesSplit(n_splits=3)
    maes, rmses = [], []
    
    plt.figure(figsize=(12, 6))
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 训练模型
        model = train_model(X_train, y_train)
        
        # 验证预测
        y_pred = model.predict(X_test)
        
        # 记录指标
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        maes.append(mae)
        rmses.append(rmse)
        
        # 绘制验证结果
        plt.plot(y_test.index, y_test, label=f'Fold {fold+1} Actual')
        plt.plot(y_test.index, y_pred, linestyle='--', label=f'Fold {fold+1} Predicted')
    
    plt.title('Cross-Validation Results')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.savefig(output_dir/"validation.png", dpi=300)
    plt.close()
    
    # 最终模型训练
    final_model = train_model(X, y)
    
    # 预测未来6个月 （可以改）
    last_data_point = X.iloc[[-1]].copy()
    forecast_steps = 6
    predictions = forecast(final_model, last_data_point, forecast_steps)
    
    # 生成预测时间索引
    last_date = feature_df.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=forecast_steps,
        freq='MS'
    )
    
    # 可视化预测结果
    plt.figure(figsize=(10, 5))
    plt.plot(feature_df.index, feature_df['总销量'], label='Historical')
    plt.plot(forecast_dates, predictions, marker='o', color='red', label='Forecast')
    plt.title('MG5 Sales Forecast') #修改为相应车型
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.savefig(output_dir/"forecast.png", dpi=300)
    plt.close()
    
    # 保存预测结果
    forecast_df = pd.DataFrame({
        'month': forecast_dates,
        'predicted_sales': predictions
    })
    forecast_df.to_csv(output_dir/"forecast_results.csv", index=False)
    
    # 输出性能报告
    print(f"模型平均性能：")
    print(f"MAE: {np.mean(maes):.1f} ± {np.std(maes):.1f}")
    print(f"RMSE: {np.mean(rmses):.1f} ± {np.std(rmses):.1f}")
    print("\n未来三个月预测：")
    print(forecast_df)

if __name__ == "__main__":
    main()