# データ生成
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ランダムシード設定（再現可能性のため）
np.random.seed(42)

# 店舗リスト
stores = ['渋谷店', '新宿店', '池袋店', '品川店', '横浜店']

# データ生成期間（3年分）
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# 各店舗の特性を定義
store_characteristics = {
    '渋谷店': {
        'base_sales': 150000,
        'weekend_multiplier': 1.4,
        'customer_per_sale': 1000,
        'group_ratio': 3.0,
        'base_labor_hours': 90
    },
    '新宿店': {
        'base_sales': 180000,
        'weekend_multiplier': 1.3,
        'customer_per_sale': 950,
        'group_ratio': 2.8,
        'base_labor_hours': 100
    },
    '池袋店': {
        'base_sales': 120000,
        'weekend_multiplier': 1.5,
        'customer_per_sale': 900,
        'group_ratio': 3.2,
        'base_labor_hours': 80
    },
    '品川店': {
        'base_sales': 100000,
        'weekend_multiplier': 1.2,
        'customer_per_sale': 1100,
        'group_ratio': 2.5,
        'base_labor_hours': 70
    },
    '横浜店': {
        'base_sales': 130000,
        'weekend_multiplier': 1.6,
        'customer_per_sale': 1050,
        'group_ratio': 3.5,
        'base_labor_hours': 85
    }
}

# 祝日リスト（2022-2024年の主要な祝日）
holidays = [
    # 2022年
    '2022-01-01', '2022-01-02', '2022-01-03', '2022-01-10', '2022-02-11', '2022-02-23',
    '2022-03-21', '2022-04-29', '2022-05-03', '2022-05-04', '2022-05-05', '2022-07-18',
    '2022-08-11', '2022-09-19', '2022-09-23', '2022-10-10', '2022-11-03', '2022-11-23',
    # 2023年
    '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-09', '2023-02-11', '2023-02-23',
    '2023-03-21', '2023-04-29', '2023-05-03', '2023-05-04', '2023-05-05', '2023-07-17',
    '2023-08-11', '2023-09-18', '2023-09-23', '2023-10-09', '2023-11-03', '2023-11-23',
    # 2024年
    '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-08', '2024-02-11', '2024-02-12',
    '2024-02-23', '2024-03-20', '2024-04-29', '2024-05-03', '2024-05-04', '2024-05-05',
    '2024-05-06', '2024-07-15', '2024-08-11', '2024-08-12', '2024-09-16', '2024-09-22',
    '2024-09-23', '2024-10-14', '2024-11-03', '2024-11-04', '2024-11-23'
]
holidays = pd.to_datetime(holidays)

# データ生成
all_data = []

for store in stores:
    char = store_characteristics[store]
    
    for date in dates:
        # 基本売上
        base_sales = char['base_sales']
        
        # 曜日効果（0:月曜 ～ 6:日曜）
        if date.weekday() == 4:  # 金曜日
            weekday_effect = char['weekend_multiplier'] * 0.9
        elif date.weekday() == 5:  # 土曜日
            weekday_effect = char['weekend_multiplier']
        elif date.weekday() == 6:  # 日曜日
            weekday_effect = char['weekend_multiplier'] * 0.95
        else:  # 平日
            weekday_effect = 1.0
        
        # イベントの設定
        event = ''
        event_effect = 1.0
        
        # 年末年始（12/28-1/3）
        if (date.month == 12 and date.day >= 28) or (date.month == 1 and date.day <= 3):
            event = '年末年始'
            event_effect = 1.5
        # ゴールデンウィーク（4/29-5/5）
        elif date.month == 4 and date.day >= 29:
            event = 'ゴールデンウィーク'
            event_effect = 1.3
        elif date.month == 5 and date.day <= 5:
            event = 'ゴールデンウィーク'
            event_effect = 1.3
        # お盆（8/13-8/16）
        elif date.month == 8 and 13 <= date.day <= 16:
            event = 'お盆'
            event_effect = 1.2
        # 祝日効果
        elif date in holidays:
            event = '祝日'
            event_effect = 1.2
        # ランダムにイベントやキャンペーンを設定（2%の確率）
        elif np.random.random() < 0.02:
            event_types = ['特別セール', 'イベント', 'キャンペーン', '地域祭り']
            event = np.random.choice(event_types)
            event_effect = np.random.uniform(1.1, 1.3)
        # 休業（0.5%の確率）
        elif np.random.random() < 0.005:
            event = '休業'
            event_effect = 0  # 売上0
        
        # 月次効果（季節性）
        month = date.month
        if month == 12:
            month_effect = 1.3
        elif month in [3, 4]:
            month_effect = 1.15
        elif month in [2, 8]:
            month_effect = 0.85
        else:
            month_effect = 1.0
        
        # 天候効果（ランダム）
        weather_effect = 0.8 if np.random.random() < 0.1 else 1.0
        
        # トレンド（3年間で緩やかに成長）
        days_from_start = (date - start_date).days
        trend_effect = 1 + (days_from_start / 1095) * 0.1
        
        # ランダム変動
        random_effect = np.random.normal(1.0, 0.1)
        
        # 最終的な売上計算
        if event_effect == 0:  # 休業の場合
            sales = 0
            customers = 0
            groups = 0
            labor_hours = 0
            sales_per_hour = 0
        else:
            sales = base_sales * weekday_effect * event_effect * month_effect * \
                    weather_effect * trend_effect * random_effect
            sales = max(sales, base_sales * 0.5)
            
            # 客数計算
            customers = int(sales / char['customer_per_sale'] * np.random.normal(1.0, 0.1))
            customers = max(customers, 10)
            
            # 組数計算
            groups = int(customers / char['group_ratio'] * np.random.normal(1.0, 0.1))
            groups = max(groups, 5)
            
            # 労働時間計算
            labor_hours = char['base_labor_hours'] * weekday_effect * event_effect * np.random.normal(1.0, 0.05)
            labor_hours = max(labor_hours, 40)
            
            # 人時売上計算
            sales_per_hour = sales / labor_hours if labor_hours > 0 else 0
        
        # データを追加
        all_data.append({
            '日付': date,
            '店舗': store,
            '売上': int(sales),
            '客数': customers,
            '組数': groups,
            '労働時間': round(labor_hours, 1),
            '人時売上': int(sales_per_hour),
            'イベント': event
        })

# DataFrameに変換
df = pd.DataFrame(all_data)

# 曜日列を追加
weekday_names = ['月', '火', '水', '木', '金', '土', '日']
df['曜日'] = df['日付'].dt.weekday.map(lambda x: weekday_names[x])

# 列の順序を整理
df = df[['日付', '曜日', '店舗', '売上', '客数', '組数', '労働時間', '人時売上', 'イベント']]

# CSVファイルとして保存
df.to_csv('restaurant_sales_test_data.csv', index=False, encoding='utf-8-sig')

# データの概要を表示
print("=== テストデータ生成完了 ===")
print(f"総レコード数: {len(df):,}件")
print(f"期間: {df['日付'].min()} ～ {df['日付'].max()}")
print(f"店舗数: {df['店舗'].nunique()}店舗")
print("\n各店舗のレコード数:")
print(df['店舗'].value_counts())
print("\n売上統計（全店舗）:")
print(df.groupby('店舗')['売上'].agg(['mean', 'min', 'max']).round(0))
print("\n最初の10件:")
print(df.head(10))
print("\n最後の10件:")
print(df.tail(10))

# イベントの統計情報
print("\nイベントの統計:")
event_stats = df[df['イベント'] != ''].groupby('イベント').agg({
    '売上': ['count', 'mean'],
    '店舗': 'nunique'
})
event_stats.columns = ['発生回数', '平均売上', '店舗数']
print(event_stats.round(0))

# 各店舗の月別平均売上も確認
monthly_avg = df.groupby(['店舗', df['日付'].dt.to_period('M')])['売上'].mean().round(0)
print("\n各店舗の月別平均売上（最新3ヶ月）:")
print(monthly_avg.groupby('店舗').tail(3))