"""
売上データの分析と予測精度改善のための診断スクリプト
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_sales_data(filepath):
    """売上データを詳細に分析して問題点を特定"""
    
    # データ読み込み
    df = pd.read_csv(filepath, encoding='utf-8')
    
    # 列名を確認して適切な列を特定
    print("=== データ列の確認 ===")
    print(f"列名: {df.columns.tolist()}")
    
    # 列名のマッピング
    date_col = None
    sales_col = None
    store_col = None
    customer_col = None
    event_col = None
    
    # 日付列の特定
    for col in ['日付', 'date', 'Date', 'DATE']:
        if col in df.columns:
            date_col = col
            break
    
    # 売上列の特定
    for col in ['売上実績', '売上', 'sales', 'Sales']:
        if col in df.columns:
            sales_col = col
            break
    
    # 店舗列の特定
    for col in ['店舗', 'store', 'Store']:
        if col in df.columns:
            store_col = col
            break
    
    # 客数列の特定
    for col in ['客数実績', '客数', 'customers', 'Customers']:
        if col in df.columns:
            customer_col = col
            break
    
    # イベント列の特定
    for col in ['イベント', 'event', 'Event']:
        if col in df.columns:
            event_col = col
            break
    
    if not date_col or not sales_col:
        print("エラー: 必要な列（日付、売上）が見つかりません")
        return None
    
    # 日付列を datetime 型に変換
    df[date_col] = pd.to_datetime(df[date_col])
    
    print("\n=== データ分析レポート ===\n")
    
    # 1. 基本統計
    print("1. 基本統計:")
    print(f"- データ期間: {df[date_col].min()} ～ {df[date_col].max()}")
    print(f"- 総レコード数: {len(df):,}件")
    if store_col:
        print(f"- 店舗数: {df[store_col].nunique()}店舗")
    print(f"- 売上平均: ¥{df[sales_col].mean():,.0f}")
    print(f"- 売上標準偏差: ¥{df[sales_col].std():,.0f}")
    print(f"- 変動係数: {(df[sales_col].std() / df[sales_col].mean()):.2%}")
    
    # 2. 欠損値チェック
    print("\n2. 欠損値の確認:")
    missing = df.isnull().sum()
    if missing.any():
        print(missing[missing > 0])
    else:
        print("欠損値なし")
    
    # 3. 異常値チェック
    print("\n3. 異常値の確認:")
    # 売上が0のレコード
    zero_sales = len(df[df[sales_col] == 0])
    print(f"- 売上0のレコード: {zero_sales}件 ({zero_sales/len(df):.1%})")
    
    # 外れ値（3σ以上）
    mean_sales = df[sales_col].mean()
    std_sales = df[sales_col].std()
    outliers = df[(df[sales_col] < mean_sales - 3*std_sales) | 
                  (df[sales_col] > mean_sales + 3*std_sales)]
    print(f"- 3σ外の外れ値: {len(outliers)}件 ({len(outliers)/len(df):.1%})")
    
    # 4. 曜日別分析
    df['曜日'] = df[date_col].dt.dayofweek
    weekday_stats = df.groupby('曜日')[sales_col].agg(['mean', 'std', 'count'])
    weekday_names = ['月', '火', '水', '木', '金', '土', '日']
    weekday_stats.index = [weekday_names[i] for i in weekday_stats.index]
    
    print("\n4. 曜日別売上統計:")
    print(weekday_stats.round(0))
    
    # 土日と平日の比較
    weekend_avg = df[df['曜日'].isin([5, 6])][sales_col].mean()
    weekday_avg = df[df['曜日'].isin([0, 1, 2, 3, 4])][sales_col].mean()
    print(f"\n- 週末平均: ¥{weekend_avg:,.0f}")
    print(f"- 平日平均: ¥{weekday_avg:,.0f}")
    print(f"- 週末/平日比: {weekend_avg/weekday_avg:.2f}倍")
    
    # 5. 月別トレンド
    df['年月'] = df[date_col].dt.to_period('M')
    monthly_trend = df.groupby('年月')[sales_col].mean()
    
    print("\n5. 月別トレンド:")
    print(f"- 最初の3ヶ月平均: ¥{monthly_trend.head(3).mean():,.0f}")
    print(f"- 最後の3ヶ月平均: ¥{monthly_trend.tail(3).mean():,.0f}")
    if len(monthly_trend) > 6:
        print(f"- 成長率: {(monthly_trend.tail(3).mean() / monthly_trend.head(3).mean() - 1):.1%}")
    
    # 6. イベント分析
    if event_col:
        print("\n6. イベント分析:")
        event_data = df[df[event_col].notna() & (df[event_col] != '')]
        if len(event_data) > 0:
            event_impact = event_data.groupby(event_col)[sales_col].agg(['mean', 'count'])
            no_event_avg = df[df[event_col].isna() | (df[event_col] == '')][sales_col].mean()
            event_impact['影響率'] = (event_impact['mean'] / no_event_avg - 1) * 100
            print(event_impact.round(0))
        else:
            print("イベントデータなし")
    
    # 7. 店舗別分析
    if store_col:
        print("\n7. 店舗別統計:")
        store_stats = df.groupby(store_col)[sales_col].agg(['mean', 'std', 'count'])
        store_stats['変動係数'] = store_stats['std'] / store_stats['mean']
        print(store_stats.round(0))
    
    # グラフ作成
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 売上推移
    ax1 = axes[0, 0]
    if store_col:
        for store in df[store_col].unique()[:3]:  # 最初の3店舗
            store_data = df[df[store_col] == store]
            store_data_monthly = store_data.groupby(store_data[date_col].dt.to_period('M'))[sales_col].mean()
            ax1.plot(store_data_monthly.index.astype(str), store_data_monthly.values, label=store, marker='o', markersize=3)
        ax1.legend()
    else:
        monthly_avg = df.groupby(df[date_col].dt.to_period('M'))[sales_col].mean()
        ax1.plot(monthly_avg.index.astype(str), monthly_avg.values, marker='o', markersize=3)
    ax1.set_title('月別売上推移')
    ax1.set_xlabel('年月')
    ax1.set_ylabel('売上')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. 曜日別売上
    ax2 = axes[0, 1]
    weekday_avg = df.groupby('曜日')[sales_col].mean()
    weekday_avg.index = [weekday_names[i] for i in weekday_avg.index]
    ax2.bar(weekday_avg.index, weekday_avg.values)
    ax2.set_title('曜日別平均売上')
    ax2.set_xlabel('曜日')
    ax2.set_ylabel('平均売上')
    
    # 3. 売上分布
    ax3 = axes[0, 2]
    ax3.hist(df[sales_col], bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(df[sales_col].mean(), color='red', linestyle='--', label='平均')
    ax3.set_title('売上分布')
    ax3.set_xlabel('売上')
    ax3.set_ylabel('頻度')
    ax3.legend()
    
    # 4. 月別売上（箱ひげ図）
    ax4 = axes[1, 0]
    df['月'] = df[date_col].dt.month
    monthly_data = []
    for m in range(1, 13):
        month_data = df[df['月'] == m][sales_col].values
        if len(month_data) > 0:
            monthly_data.append(month_data)
        else:
            monthly_data.append([0])
    ax4.boxplot(monthly_data, labels=range(1, 13))
    ax4.set_title('月別売上分布')
    ax4.set_xlabel('月')
    ax4.set_ylabel('売上')
    
    # 5. 自己相関
    ax5 = axes[1, 1]
    from pandas.plotting import autocorrelation_plot
    if store_col:
        sample_store = df[store_col].unique()[0]
        sample_data = df[df[store_col] == sample_store].sort_values(date_col)[sales_col]
    else:
        sample_data = df.sort_values(date_col)[sales_col]
    autocorrelation_plot(sample_data, ax=ax5)
    ax5.set_title('自己相関')
    
    # 6. 散布図（客数 vs 売上）
    ax6 = axes[1, 2]
    if customer_col:
        ax6.scatter(df[customer_col], df[sales_col], alpha=0.5)
        ax6.set_xlabel('客数')
        ax6.set_ylabel('売上')
        ax6.set_title('客数 vs 売上')
    else:
        ax6.text(0.5, 0.5, '客数データなし', ha='center', va='center')
        ax6.set_title('客数 vs 売上')
    
    plt.tight_layout()
    plt.show()
    
    # 改善提案
    print("\n=== 予測精度改善のための提案 ===")
    
    suggestions = []
    
    # 1. データ量チェック
    if len(df) < 365:
        suggestions.append("データ量不足: 最低1年分のデータが必要です。")
    
    # 2. 変動係数チェック
    cv = df[sales_col].std() / df[sales_col].mean()
    if cv > 0.5:
        suggestions.append(f"売上変動が大きい（CV={cv:.2f}）: 外れ値の除去や変換を検討してください。")
    
    # 3. 曜日パターン
    if weekend_avg / weekday_avg > 1.5 or weekend_avg / weekday_avg < 0.7:
        suggestions.append("曜日による差が大きい: 曜日別モデルの構築を検討してください。")
    
    # 4. トレンド
    if len(monthly_trend) > 6:
        growth_rate = abs(monthly_trend.tail(3).mean() / monthly_trend.head(3).mean() - 1)
        if growth_rate > 0.2:
            suggestions.append("明確なトレンドあり: トレンド成分を明示的にモデルに組み込んでください。")
    
    # 5. 季節性
    monthly_avg = df.groupby(df[date_col].dt.month)[sales_col].mean()
    if len(monthly_avg) == 12 and (monthly_avg.max() / monthly_avg.min()) > 1.3:
        suggestions.append("季節性あり: 月別ダミー変数や周期的特徴量を追加してください。")
    
    # 6. イベント
    if event_col:
        event_ratio = len(df[df[event_col].notna() & (df[event_col] != '')]) / len(df)
        if event_ratio < 0.05:
            suggestions.append("イベントデータ不足: もっと多くのイベント情報を記録してください。")
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
    
    # 推奨モデル
    print("\n推奨モデル:")
    if cv > 0.5:
        print("- LightGBM or XGBoost（外れ値に強い）")
    if weekend_avg / weekday_avg > 1.3:
        print("- 曜日別の個別モデル")
    if len(monthly_trend) > 6:
        growth_rate = abs(monthly_trend.tail(3).mean() / monthly_trend.head(3).mean() - 1)
        if growth_rate > 0.1:
            print("- Prophet（トレンドと季節性を自動処理）")
    
    return df

# 使用例
if __name__ == "__main__":
    # ファイルパスを指定して実行
    filepath = 'restaurant_sales_test_data.csv'
    df_analyzed = analyze_sales_data(filepath)