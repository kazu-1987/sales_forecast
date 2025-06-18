"""
飲食店売上予測アプリケーション - Streamlit版
実行方法: streamlit run app.py
必要なライブラリ: pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import os
import pickle

# ページ設定
st.set_page_config(
    page_title="飲食店売上予測システム",
    page_icon="📊",
    layout="wide"
)

# タイトル
st.title("🍴 飲食店売上予測システム")
st.markdown("---")

# サイドバーの設定
st.sidebar.header("📁 データ管理")

# セッション状態の初期化
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None

def load_and_process_data(uploaded_file):
    """データの読み込みと前処理"""
    try:
        # CSVファイルを読み込み
        data = pd.read_csv(uploaded_file, encoding='utf-8')
        
        # 日付列を datetime 型に変換
        date_columns = ['日付', 'date', 'Date', 'DATE']
        for col in date_columns:
            if col in data.columns:
                data['date'] = pd.to_datetime(data[col])
                if col != 'date':
                    data = data.drop(col, axis=1)
                break
        
        # 列名を英語に統一
        column_mapping = {
            '店舗': 'store',
            '店舗名': 'store',
            '売上': 'sales',
            '売上高': 'sales',
            '売上実績': 'sales',  # 新しい列名に対応
            '客数': 'customers',
            '客数実績': 'customers',  # 新しい列名に対応
            '組数': 'groups',
            '労働時間': 'labor_hours',
            '人時売上': 'sales_per_hour',
            '曜日': 'weekday_jp',
            'イベント': 'event'
        }
        
        for jp_col, en_col in column_mapping.items():
            if jp_col in data.columns:
                data[en_col] = data[jp_col]
                if jp_col != en_col:
                    data = data.drop(jp_col, axis=1)
        
        # 店舗名がない場合の処理
        if 'store' not in data.columns:
            data['store'] = 'デフォルト店舗'
        
        # イベント列がない場合は空文字で作成
        if 'event' not in data.columns:
            data['event'] = ''
        
        # イベント列のNaNを空文字に変換
        data['event'] = data['event'].fillna('')
        
        return data
        
    except Exception as e:
        st.error(f"データ読み込みエラー: {str(e)}")
        return None

def create_features(data):
    """特徴量を作成"""
    data = data.copy()
    
    # 基本的な時間特徴量
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek
    data['day_of_year'] = data['date'].dt.dayofyear
    data['week_of_year'] = data['date'].dt.isocalendar().week
    data['quarter'] = data['date'].dt.quarter
    data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
    data['is_month_start'] = data['date'].dt.is_month_start.astype(int)
    data['is_month_end'] = data['date'].dt.is_month_end.astype(int)
    
    # 曜日ごとのダミー変数を作成（土日の売上パターンを正確に捉えるため）
    for i in range(7):
        data[f'is_weekday_{i}'] = (data['dayofweek'] == i).astype(int)
    
    # 金曜日、土曜日、日曜日の特別フラグ
    data['is_friday'] = (data['dayofweek'] == 4).astype(int)
    data['is_saturday'] = (data['dayofweek'] == 5).astype(int)
    data['is_sunday'] = (data['dayofweek'] == 6).astype(int)
    
    # イベント関連の特徴量を作成
    if 'event' in data.columns:
        # イベントフラグ（イベントがあるかどうか）
        data['has_event'] = (data['event'] != '').astype(int)
        
        # 主要イベントのダミー変数を作成
        # イベント名を統一（大文字小文字、スペースを無視）
        data['event_normalized'] = data['event'].str.upper().str.replace(' ', '').str.replace('　', '')
        
        # 主要イベントのカテゴリ
        event_categories = {
            'year_end': ['年末年始', '年末', '年始', '正月', 'YEAREND', 'NEWYEAR'],
            'golden_week': ['ゴールデンウィーク', 'GW', 'GOLDENWEEK'],
            'obon': ['お盆', 'OBON', '盆'],
            'holiday': ['休業', '休み', 'CLOSED', 'HOLIDAY'],
            'special_sale': ['セール', 'キャンペーン', 'SALE', 'CAMPAIGN'],
            'event': ['イベント', 'EVENT', '祭り', 'FESTIVAL']
        }
        
        # 各カテゴリのフラグを作成
        for category, keywords in event_categories.items():
            data[f'event_{category}'] = 0
            for keyword in keywords:
                data.loc[data['event_normalized'].str.contains(keyword, na=False), f'event_{category}'] = 1
    
    # 店舗ごとの移動平均と曜日別統計
    for store in data['store'].unique():
        store_mask = data['store'] == store
        store_data = data[store_mask].sort_values('date')
        
        # 7日移動平均
        data.loc[store_mask, 'sales_ma7'] = store_data['sales'].rolling(window=7, min_periods=1).mean()
        data.loc[store_mask, 'sales_ma30'] = store_data['sales'].rolling(window=30, min_periods=1).mean()
        
        # 曜日別の売上統計を特徴量として追加
        weekday_stats = store_data.groupby('dayofweek')['sales'].agg(['mean', 'std', 'median'])
        for dow in range(7):
            if dow in weekday_stats.index:
                data.loc[store_mask & (data['dayofweek'] == dow), 'weekday_sales_mean'] = weekday_stats.loc[dow, 'mean']
                data.loc[store_mask & (data['dayofweek'] == dow), 'weekday_sales_std'] = weekday_stats.loc[dow, 'std']
            else:
                data.loc[store_mask & (data['dayofweek'] == dow), 'weekday_sales_mean'] = store_data['sales'].mean()
                data.loc[store_mask & (data['dayofweek'] == dow), 'weekday_sales_std'] = store_data['sales'].std()
        
        if 'customers' in data.columns:
            data.loc[store_mask, 'customers_ma7'] = store_data['customers'].rolling(window=7, min_periods=1).mean()
    
    return data

def train_model(store_data, store_name):
    """モデルの学習"""
    # 特徴量リスト
    feature_columns = ['year', 'month', 'day', 'dayofweek', 'day_of_year', 
                      'week_of_year', 'quarter', 'is_weekend', 
                      'is_month_start', 'is_month_end',
                      'is_friday', 'is_saturday', 'is_sunday']
    
    # 曜日ダミー変数を追加
    for i in range(7):
        feature_columns.append(f'is_weekday_{i}')
    
    # 曜日別統計特徴量を追加
    if 'weekday_sales_mean' in store_data.columns:
        feature_columns.extend(['weekday_sales_mean', 'weekday_sales_std'])
    
    # イベント関連の特徴量を追加
    event_features = ['has_event', 'event_year_end', 'event_golden_week', 
                     'event_obon', 'event_holiday', 'event_special_sale', 'event_event']
    for col in event_features:
        if col in store_data.columns:
            feature_columns.append(col)
    
    # オプション特徴量の追加
    for col in ['customers', 'groups', 'labor_hours', 'sales_per_hour', 
                'sales_ma7', 'sales_ma30', 'customers_ma7']:
        if col in store_data.columns and not store_data[col].isna().all():
            feature_columns.append(col)
    
    # データの準備
    store_data = store_data.sort_values('date')
    store_data = store_data.dropna(subset=['sales'] + [col for col in feature_columns if col in store_data.columns])
    
    # 使用可能な特徴量のみを選択
    available_features = [col for col in feature_columns if col in store_data.columns]
    
    X = store_data[available_features]
    y = store_data['sales']
    
    # 時系列分割
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # モデル学習（パラメータを調整）
    from sklearn.ensemble import GradientBoostingRegressor
    
    # ランダムフォレストの代わりに勾配ブースティングを試す
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )
    
    # 代替案：より深いランダムフォレスト
    # model = RandomForestRegressor(
    #     n_estimators=200,
    #     max_depth=15,
    #     min_samples_split=10,
    #     min_samples_leaf=5,
    #     random_state=42,
    #     n_jobs=-1
    # )
    
    model.fit(X_train, y_train)
    
    # 評価
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)
    
    # 曜日別の予測精度を確認
    test_data = store_data.iloc[split_index:].copy()
    test_data['predicted'] = y_pred
    weekday_accuracy = test_data.groupby('dayofweek').agg({
        'sales': 'mean',
        'predicted': 'mean'
    })
    
    # 結果を保存
    result = {
        'model': model,
        'feature_columns': available_features,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'test_dates': store_data.iloc[split_index:]['date'].values,
        'test_actual': y_test.values,
        'test_pred': y_pred,
        'feature_importance': pd.DataFrame({
            'feature': available_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False),
        'event_impact': {},
        'weekday_accuracy': weekday_accuracy,
        'weekday_stats': store_data.groupby('dayofweek')['sales'].agg(['mean', 'std', 'count'])
    }
    
    # イベントの影響を分析
    if 'has_event' in available_features:
        event_data = store_data[store_data['has_event'] == 1]['sales'].mean()
        no_event_data = store_data[store_data['has_event'] == 0]['sales'].mean()
        if not pd.isna(event_data) and not pd.isna(no_event_data):
            result['event_impact']['overall'] = (event_data / no_event_data - 1) * 100
    
    return result

def make_forecast(model_info, store_data, days):
    """予測を実行"""
    model = model_info['model']
    feature_columns = model_info['feature_columns']
    
    last_date = store_data['date'].max()
    last_row = store_data[store_data['date'] == last_date].iloc[0]
    
    # 曜日別の売上統計を取得
    weekday_stats = model_info.get('weekday_stats', store_data.groupby('dayofweek')['sales'].agg(['mean', 'std', 'count']))
    
    future_dates = []
    future_features = []
    
    for i in range(1, days + 1):
        future_date = last_date + timedelta(days=i)
        future_dates.append(future_date)
        
        # 基本特徴量を作成
        features = {
            'year': future_date.year,
            'month': future_date.month,
            'day': future_date.day,
            'dayofweek': future_date.weekday(),
            'day_of_year': future_date.timetuple().tm_yday,
            'week_of_year': future_date.isocalendar().week,
            'quarter': (future_date.month - 1) // 3 + 1,
            'is_weekend': 1 if future_date.weekday() >= 5 else 0,
            'is_month_start': 1 if future_date.day == 1 else 0,
            'is_month_end': 1 if (future_date + timedelta(days=1)).day == 1 else 0,
            'is_friday': 1 if future_date.weekday() == 4 else 0,
            'is_saturday': 1 if future_date.weekday() == 5 else 0,
            'is_sunday': 1 if future_date.weekday() == 6 else 0
        }
        
        # 曜日ダミー変数
        for j in range(7):
            features[f'is_weekday_{j}'] = 1 if future_date.weekday() == j else 0
        
        # 曜日別統計特徴量
        dow = future_date.weekday()
        if 'weekday_sales_mean' in feature_columns:
            if dow in weekday_stats.index:
                features['weekday_sales_mean'] = weekday_stats.loc[dow, 'mean']
                features['weekday_sales_std'] = weekday_stats.loc[dow, 'std']
            else:
                features['weekday_sales_mean'] = store_data['sales'].mean()
                features['weekday_sales_std'] = store_data['sales'].std()
        
        # イベント関連の特徴量（デフォルトは0）
        event_features = ['has_event', 'event_year_end', 'event_golden_week', 
                         'event_obon', 'event_holiday', 'event_special_sale', 'event_event']
        for ef in event_features:
            if ef in feature_columns:
                features[ef] = 0
        
        # その他の特徴量を追加
        for col in feature_columns:
            if col not in features:
                if col in last_row.index and not pd.isna(last_row[col]):
                    features[col] = last_row[col]
                elif col in ['sales_ma7', 'sales_ma30']:
                    # 移動平均は最後の値を使用（NaNチェック付き）
                    if col in last_row.index and not pd.isna(last_row[col]):
                        features[col] = last_row[col]
                    else:
                        features[col] = store_data['sales'].tail(30).mean()
                elif col == 'customers_ma7':
                    if 'customers' in store_data.columns:
                        features[col] = store_data['customers'].tail(7).mean()
                    else:
                        features[col] = 0
                elif col in store_data.columns:
                    # その他の列は平均値を使用
                    mean_val = store_data[col].mean()
                    features[col] = mean_val if not pd.isna(mean_val) else 0
                else:
                    # デフォルト値
                    features[col] = 0
        
        # NaNチェックと置換
        feature_vector = []
        for col in feature_columns:
            val = features.get(col, 0)
            if pd.isna(val):
                val = 0
            feature_vector.append(val)
        
        future_features.append(feature_vector)
    
    # 売上予測
    predictions = model.predict(future_features)
    
    # 結果をDataFrameに
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'predicted_sales': predictions
    })
    
    # 他の指標も予測（過去データから相関を使用）
    # 客数予測
    if 'customers' in store_data.columns and 'sales' in store_data.columns:
        # 売上が0でないデータのみを使用
        valid_data = store_data[store_data['sales'] > 0]
        if len(valid_data) > 0:
            avg_customer_per_sale = (valid_data['customers'] / valid_data['sales']).mean()
            forecast_df['predicted_customers'] = (forecast_df['predicted_sales'] * avg_customer_per_sale).round().astype(int)
        else:
            forecast_df['predicted_customers'] = 0
    else:
        forecast_df['predicted_customers'] = 0
    
    # 労働時間予測（曜日別の傾向を使用）
    forecast_df['dayofweek'] = [d.weekday() for d in future_dates]
    
    # 曜日別の売上と労働時間の関係から推定
    if 'labor_hours' in store_data.columns:
        weekday_labor = store_data.groupby('dayofweek')['labor_hours'].mean()
        forecast_df['predicted_labor_hours'] = forecast_df['dayofweek'].map(weekday_labor)
    else:
        # 労働時間データがない場合は、売上に基づいて推定
        base_hours = 80
        forecast_df['predicted_labor_hours'] = base_hours * (forecast_df['predicted_sales'] / store_data['sales'].mean())
    
    forecast_df['predicted_labor_hours'] = forecast_df['predicted_labor_hours'].fillna(85).round(1)
    
    # 人時売上予測
    forecast_df['predicted_sales_per_hour'] = (forecast_df['predicted_sales'] / forecast_df['predicted_labor_hours']).round().astype(int)
    
    # 不要な列を削除
    if 'dayofweek' in forecast_df.columns:
        forecast_df = forecast_df.drop('dayofweek', axis=1)
    
    return forecast_df

# メインアプリケーション
# 1. データアップロード
uploaded_file = st.sidebar.file_uploader(
    "CSVファイルをアップロード",
    type=['csv'],
    help="日付、店舗、売上、客数、組数、労働時間、人時売上などのデータを含むCSVファイル"
)

if uploaded_file is not None:
    # データを読み込み
    data = load_and_process_data(uploaded_file)
    
    if data is not None:
        # 特徴量を作成
        data = create_features(data)
        st.session_state.data = data
        
        # データ概要を表示
        st.sidebar.success(f"✅ データ読み込み完了: {len(data):,}件")
        st.sidebar.info(f"📅 期間: {data['date'].min().strftime('%Y-%m-%d')} ～ {data['date'].max().strftime('%Y-%m-%d')}")
        st.sidebar.info(f"🏪 店舗数: {data['store'].nunique()}店舗")

# データが読み込まれている場合
if st.session_state.data is not None:
    data = st.session_state.data
    
    # タブを作成
    tab1, tab2, tab3, tab4 = st.tabs(["📊 データ概要", "🤖 モデル学習", "🔮 売上予測", "📈 分析レポート"])
    
    # タブ1: データ概要
    with tab1:
        st.header("データ概要")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("総データ数", f"{len(data):,}件")
        with col2:
            st.metric("店舗数", f"{data['store'].nunique()}店舗")
        with col3:
            st.metric("データ期間", f"{(data['date'].max() - data['date'].min()).days}日間")
        
        # 店舗選択
        selected_store = st.selectbox("店舗を選択", data['store'].unique())
        store_data = data[data['store'] == selected_store]
        
        # 売上推移グラフ
        st.subheader(f"{selected_store}の売上推移")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=store_data['date'],
            y=store_data['sales'],
            mode='lines',
            name='売上',
            line=dict(color='blue', width=1)
        ))
        fig.update_layout(
            xaxis_title="日付",
            yaxis_title="売上（円）",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # データサンプル表示
        st.subheader("データサンプル（最新10件）")
        display_cols = ['date', 'sales', 'customers']
        if 'groups' in store_data.columns:
            display_cols.append('groups')
        if 'labor_hours' in store_data.columns:
            display_cols.append('labor_hours')
        if 'event' in store_data.columns:
            display_cols.append('event')
        
        st.dataframe(store_data.tail(10)[display_cols])
    
    # タブ2: モデル学習
    with tab2:
        st.header("モデル学習")
        
        # 学習する店舗を選択
        stores_to_train = st.multiselect(
            "学習する店舗を選択",
            data['store'].unique(),
            default=data['store'].unique()
        )
        
        if st.button("🚀 モデルを学習", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, store in enumerate(stores_to_train):
                status_text.text(f"学習中: {store}")
                store_data = data[data['store'] == store]
                
                if len(store_data) >= 30:
                    model_info = train_model(store_data, store)
                    st.session_state.models[store] = model_info
                    
                    # 結果を表示
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{store} - MAE", f"¥{model_info['mae']:,.0f}")
                    with col2:
                        st.metric(f"{store} - MAPE", f"{model_info['mape']:.1f}%")
                    with col3:
                        st.metric(f"{store} - R²", f"{model_info['r2']:.3f}")
                    
                    # 曜日別精度を表示
                    if 'weekday_accuracy' in model_info:
                        with st.expander(f"{store} の曜日別予測精度"):
                            weekday_names = ['月', '火', '水', '木', '金', '土', '日']
                            accuracy_df = model_info['weekday_accuracy'].copy()
                            accuracy_df.index = [weekday_names[i] for i in accuracy_df.index]
                            accuracy_df.columns = ['実績平均', '予測平均']
                            accuracy_df['差異(%)'] = ((accuracy_df['予測平均'] / accuracy_df['実績平均'] - 1) * 100).round(1)
                            st.dataframe(accuracy_df)
                    
                    # 曜日別精度を表示
                    if 'weekday_accuracy' in model_info:
                        with st.expander(f"{store} の曜日別予測精度"):
                            weekday_names = ['月', '火', '水', '木', '金', '土', '日']
                            accuracy_df = model_info['weekday_accuracy'].copy()
                            accuracy_df.index = [weekday_names[i] for i in accuracy_df.index]
                            accuracy_df.columns = ['実績平均', '予測平均']
                            accuracy_df['差異(%)'] = ((accuracy_df['予測平均'] / accuracy_df['実績平均'] - 1) * 100).round(1)
                            st.dataframe(accuracy_df)
                    with col1:
                        st.metric(f"{store} - MAE", f"¥{model_info['mae']:,.0f}")
                    with col2:
                        st.metric(f"{store} - MAPE", f"{model_info['mape']:.1f}%")
                    with col3:
                        st.metric(f"{store} - R²", f"{model_info['r2']:.3f}")
                else:
                    st.warning(f"{store}のデータが不足しています（{len(store_data)}件）")
                
                progress_bar.progress((i + 1) / len(stores_to_train))
            
            status_text.text("✅ 学習完了！")
        
        # 学習済みモデルの情報
        if st.session_state.models:
            st.subheader("学習済みモデル")
            model_summary = []
            for store, info in st.session_state.models.items():
                model_summary.append({
                    '店舗': store,
                    'MAE': f"¥{info['mae']:,.0f}",
                    'MAPE': f"{info['mape']:.1f}%",
                    'R²': f"{info['r2']:.3f}"
                })
            st.dataframe(pd.DataFrame(model_summary))
    
    # タブ3: 売上予測
    with tab3:
        st.header("売上予測")
        
        if not st.session_state.models:
            st.warning("⚠️ まずモデルを学習してください")
        else:
            col1, col2 = st.columns(2)
            with col1:
                forecast_store = st.selectbox(
                    "予測する店舗",
                    list(st.session_state.models.keys())
                )
            with col2:
                forecast_days = st.slider("予測日数", 7, 90, 30)
            
            if st.button("📈 予測実行", type="primary"):
                # 予測を実行
                store_data = data[data['store'] == forecast_store]
                model_info = st.session_state.models[forecast_store]
                forecast_df = make_forecast(model_info, store_data, forecast_days)
                
                # 結果を保存
                st.session_state.forecast_results = {
                    'store': forecast_store,
                    'forecast': forecast_df
                }
                
                # 予測結果のサマリー
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("予測期間合計売上", f"¥{forecast_df['predicted_sales'].sum():,.0f}")
                with col2:
                    st.metric("日次平均売上", f"¥{forecast_df['predicted_sales'].mean():,.0f}")
                with col3:
                    st.metric("期間中の平均人時売上", f"¥{forecast_df['predicted_sales_per_hour'].mean():,.0f}")
                
                # グラフ表示
                st.subheader("売上予測グラフ")
                
                # 実績と予測を結合してプロット
                recent_actual = store_data.tail(60)
                
                fig = go.Figure()
                
                # 実績
                fig.add_trace(go.Scatter(
                    x=recent_actual['date'],
                    y=recent_actual['sales'],
                    mode='lines+markers',
                    name='実績売上',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ))
                
                # 予測
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['predicted_sales'],
                    mode='lines+markers',
                    name='予測売上',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=4)
                ))
                
                fig.update_layout(
                    xaxis_title="日付",
                    yaxis_title="売上（円）",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 日毎の詳細予測表示
                st.subheader("日毎予測詳細")
                
                # 表示用のDataFrameを作成
                display_df = forecast_df.copy()
                
                # 曜日名を追加
                weekday_names = ['月', '火', '水', '木', '金', '土', '日']
                display_df['曜日'] = display_df['date'].dt.dayofweek.apply(lambda x: weekday_names[x])
                
                # 日付を曜日付きでフォーマット
                display_df['日付'] = display_df['date'].dt.strftime('%Y-%m-%d') + ' (' + display_df['曜日'] + ')'
                display_df['店舗'] = forecast_store
                display_df['予測売上'] = display_df['predicted_sales'].apply(lambda x: f"¥{x:,.0f}")
                display_df['予測客数'] = display_df['predicted_customers']
                display_df['予測労働時間'] = display_df['predicted_labor_hours']
                display_df['予測人時'] = display_df['predicted_sales_per_hour'].apply(lambda x: f"¥{x:,.0f}")
                
                # 指定された順番で表示
                display_columns = ['日付', '店舗', '予測売上', '予測客数', '予測労働時間', '予測人時']
                display_df = display_df[display_columns]
                
                # データフレームを表示（スクロール可能）
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
                
                # 予測データのダウンロード
                # エクスポート用のDataFrameを作成（フォーマット前の数値データ）
                export_df = forecast_df.copy()
                weekday_names_export = ['月', '火', '水', '木', '金', '土', '日']
                export_df['曜日'] = export_df['date'].dt.dayofweek.apply(lambda x: weekday_names_export[x])
                export_df['日付'] = export_df['date'].dt.strftime('%Y-%m-%d')
                export_df['店舗'] = forecast_store
                export_df = export_df.rename(columns={
                    'predicted_sales': '予測売上',
                    'predicted_customers': '予測客数',
                    'predicted_labor_hours': '予測労働時間',
                    'predicted_sales_per_hour': '予測人時'
                })
                export_df = export_df[['日付', '曜日', '店舗', '予測売上', '予測客数', '予測労働時間', '予測人時']]
                
                csv = export_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 予測データをダウンロード",
                    data=csv,
                    file_name=f'{forecast_store}_forecast_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )
    
    # タブ4: 分析レポート
    with tab4:
        st.header("分析レポート")
        
        if st.session_state.models:
            # 特徴量重要度
            st.subheader("特徴量の重要度")
            
            selected_store_report = st.selectbox(
                "分析する店舗",
                list(st.session_state.models.keys()),
                key="report_store"
            )
            
            model_info = st.session_state.models[selected_store_report]
            importance_df = model_info['feature_importance'].head(10)
            
            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title=f"{selected_store_report}の特徴量重要度（上位10）"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # 全店舗比較
            st.subheader("全店舗パフォーマンス比較")
            
            comparison_data = []
            for store, info in st.session_state.models.items():
                comparison_data.append({
                    '店舗': store,
                    'MAE': info['mae'],
                    'MAPE': info['mape'],
                    'R²': info['r2']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # MAPEでソートして棒グラフ
            comparison_df = comparison_df.sort_values('MAPE')
            
            fig = px.bar(
                comparison_df,
                x='店舗',
                y='MAPE',
                title="店舗別予測精度（MAPE: 低いほど良い）",
                color='MAPE',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # モデル保存
            st.subheader("モデルの保存")
            if st.button("💾 全モデルを保存"):
                # モデルをpickleで保存
                with open('sales_forecast_models.pkl', 'wb') as f:
                    pickle.dump(st.session_state.models, f)
                st.success("✅ モデルを保存しました（sales_forecast_models.pkl）")

else:
    # データがアップロードされていない場合
    st.info("👈 左のサイドバーからCSVファイルをアップロードしてください")
    
    # サンプルデータの説明
    st.subheader("必要なデータ形式")
    st.markdown("""
    CSVファイルには以下の列が必要です：
    - **日付**: 売上日（YYYY-MM-DD形式）
    - **店舗**: 店舗名
    - **売上実績** または **売上**: 売上金額
    - **客数実績** または **客数**: 来店客数（オプション）
    - **イベント**: イベント名（オプション）
    
    ※ その他のオプション列：組数、労働時間、人時売上
    """)
    
    # サンプルデータ表示
    sample_data = pd.DataFrame({
        '日付': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-12-31'],
        '店舗': ['渋谷店', '渋谷店', '渋谷店', '渋谷店'],
        '売上実績': [150000, 180000, 165000, 250000],
        '客数実績': [150, 180, 165, 300],
        'イベント': ['年末年始', '', '', '年末年始']
    })
    st.dataframe(sample_data)