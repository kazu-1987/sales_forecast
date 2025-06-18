"""
飲食店売上予測アプリケーション - Prophet版
実行方法: streamlit run app_prophet.py
必要なライブラリ: pip install streamlit pandas numpy prophet plotly matplotlib
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import warnings
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="飲食店売上予測システム - Prophet版",
    page_icon="📊",
    layout="wide"
)

# タイトル
st.title("🍴 飲食店売上予測システム (Prophet版)")
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
            '売上実績': 'sales',
            '客数': 'customers',
            '客数実績': 'customers',
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

def prepare_prophet_data(store_data, include_regressors=True):
    """Prophet用のデータ準備"""
    # Prophet形式のデータフレームを作成
    prophet_df = pd.DataFrame({
        'ds': store_data['date'],
        'y': store_data['sales']
    })
    
    # 外れ値の処理（オプション）
    # 3σを超える値をcapとfloorで制限
    mean_sales = prophet_df['y'].mean()
    std_sales = prophet_df['y'].std()
    prophet_df['cap'] = mean_sales + 3 * std_sales
    prophet_df['floor'] = max(0, mean_sales - 3 * std_sales)
    
    if include_regressors:
        # 追加の回帰変数を準備
        # イベントフラグ
        if 'event' in store_data.columns:
            prophet_df['has_event'] = (store_data['event'] != '').astype(int)
            
            # 主要イベントのダミー変数
            prophet_df['is_year_end'] = store_data['event'].str.contains('年末年始|年末|年始', na=False).astype(int)
            prophet_df['is_golden_week'] = store_data['event'].str.contains('ゴールデンウィーク|GW', na=False).astype(int)
            prophet_df['is_obon'] = store_data['event'].str.contains('お盆|盆', na=False).astype(int)
            prophet_df['is_holiday'] = store_data['event'].str.contains('休業|休み', na=False).astype(int)
        
        # 客数（あれば）
        if 'customers' in store_data.columns:
            prophet_df['customers'] = store_data['customers']
    
    return prophet_df

def train_prophet_model(prophet_df, store_name):
    """Prophetモデルの学習"""
    # モデルの初期化
    model = Prophet(
        growth='linear',  # or 'logistic'
        changepoint_prior_scale=0.05,  # トレンドの柔軟性
        seasonality_prior_scale=10.0,  # 季節性の強さ
        holidays_prior_scale=10.0,  # 祝日効果の強さ
        seasonality_mode='multiplicative',  # 乗法的季節性
        interval_width=0.95,  # 予測区間
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    
    # 日本の祝日を追加（簡易版）
    holidays = pd.DataFrame({
        'holiday': 'japanese_holiday',
        'ds': pd.to_datetime([
            '2022-01-01', '2022-01-02', '2022-01-03', '2022-05-03', '2022-05-04', '2022-05-05',
            '2023-01-01', '2023-01-02', '2023-01-03', '2023-05-03', '2023-05-04', '2023-05-05',
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-05-03', '2024-05-04', '2024-05-05',
            '2025-01-01', '2025-01-02', '2025-01-03', '2025-05-03', '2025-05-04', '2025-05-05'
        ])
    })
    model.holidays = holidays
    
    # 回帰変数の追加
    if 'has_event' in prophet_df.columns:
        model.add_regressor('has_event')
    if 'is_year_end' in prophet_df.columns:
        model.add_regressor('is_year_end')
    if 'is_golden_week' in prophet_df.columns:
        model.add_regressor('is_golden_week')
    if 'is_obon' in prophet_df.columns:
        model.add_regressor('is_obon')
    
    # モデルの学習
    model.fit(prophet_df)
    
    # 検証用の予測
    # 最後の20%をテストデータとして使用
    split_index = int(len(prophet_df) * 0.8)
    train_df = prophet_df[:split_index]
    test_df = prophet_df[split_index:]
    
    # 評価指標の初期化
    mae = 0
    mape = 0
    rmse = 0
    r2 = 0
    weekday_accuracy = pd.DataFrame()
    
    # テストデータ期間の予測（テストデータが十分にある場合のみ）
    if len(test_df) > 7:  # 最低1週間分のデータがある場合
        try:
            # 全期間のデータフレームを作成
            future_all = model.make_future_dataframe(periods=0)
            
            # 回帰変数を追加
            for col in ['has_event', 'is_year_end', 'is_golden_week', 'is_obon']:
                if col in prophet_df.columns:
                    future_all = future_all.merge(
                        prophet_df[['ds', col]], 
                        on='ds', 
                        how='left'
                    ).fillna(0)
            
            # 予測実行
            forecast_all = model.predict(future_all)
            
            # テスト期間のデータを抽出
            test_forecast = forecast_all[forecast_all['ds'].isin(test_df['ds'])].copy()
            test_actual = test_df.merge(test_forecast[['ds', 'yhat']], on='ds', how='left')
            
            # 精度評価
            actual = test_actual['y'].values
            predicted = test_actual['yhat'].values
            
            # NaNを除外
            mask = ~(np.isnan(actual) | np.isnan(predicted))
            actual = actual[mask]
            predicted = predicted[mask]
            
            if len(actual) > 0:
                mae = np.mean(np.abs(actual - predicted))
                mape = np.mean(np.abs((actual - predicted) / (actual + 1))) * 100  # +1で0除算を防ぐ
                rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                
                # R²スコア
                if len(actual) > 1:
                    from sklearn.metrics import r2_score
                    r2 = r2_score(actual, predicted)
                
                # 曜日別精度
                test_actual['dayofweek'] = pd.to_datetime(test_actual['ds']).dt.dayofweek
                weekday_accuracy = test_actual.groupby('dayofweek').agg({
                    'y': 'mean',
                    'yhat': 'mean'
                }).rename(columns={'yhat': 'predicted'})
        except Exception as e:
            st.warning(f"テストデータでの評価中にエラーが発生しました: {str(e)}")
    
    return {
        'model': model,
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'r2': r2,
        'weekday_accuracy': weekday_accuracy,
        'prophet_df': prophet_df
    }

def make_prophet_forecast(model_info, store_data, total_days):
    """Prophetによる予測（過去期間も含む）"""
    model = model_info['model']
    prophet_df = model_info['prophet_df']
    
    # 将来の日付を作成（過去のデータも含めて全期間をカバー）
    # Prophetは学習データの最初から予測するため、make_future_dataframeで全期間を生成
    future = model.make_future_dataframe(periods=total_days, freq='D', include_history=True)
    
    # 将来の回帰変数を設定
    for col in ['has_event', 'is_year_end', 'is_golden_week', 'is_obon']:
        if col in prophet_df.columns:
            # 既存データから回帰変数を取得
            existing_values = prophet_df[['ds', col]]
            future = future.merge(existing_values, on='ds', how='left')
            # 将来の値は0で埋める
            future[col] = future[col].fillna(0)
    
    # 予測実行
    forecast = model.predict(future)
    
    # 結果を整形
    forecast_df = pd.DataFrame({
        'date': forecast['ds'],
        'predicted_sales': forecast['yhat'],
        'lower_bound': forecast['yhat_lower'],
        'upper_bound': forecast['yhat_upper']
    })
    
    # 客数予測
    if 'customers' in store_data.columns and 'sales' in store_data.columns:
        valid_data = store_data[store_data['sales'] > 0]
        if len(valid_data) > 0:
            avg_customer_per_sale = (valid_data['customers'] / valid_data['sales']).mean()
            forecast_df['predicted_customers'] = (forecast_df['predicted_sales'] * avg_customer_per_sale).round().astype(int)
        else:
            forecast_df['predicted_customers'] = 0
    else:
        forecast_df['predicted_customers'] = 0
    
    # 労働時間予測
    forecast_df['dayofweek'] = forecast_df['date'].dt.dayofweek
    if 'labor_hours' in store_data.columns:
        weekday_labor = store_data.groupby(store_data['date'].dt.dayofweek)['labor_hours'].mean()
        forecast_df['predicted_labor_hours'] = forecast_df['dayofweek'].map(weekday_labor)
    else:
        base_hours = 80
        forecast_df['predicted_labor_hours'] = base_hours * (forecast_df['predicted_sales'] / store_data['sales'].mean())
    
    forecast_df['predicted_labor_hours'] = forecast_df['predicted_labor_hours'].fillna(85).round(1)
    
    # 人時売上予測
    forecast_df['predicted_sales_per_hour'] = (forecast_df['predicted_sales'] / forecast_df['predicted_labor_hours']).round().astype(int)
    
    # 不要な列を削除
    forecast_df = forecast_df.drop('dayofweek', axis=1)
    
    return forecast_df, forecast

# メインアプリケーション
# 1. データアップロード
uploaded_file = st.sidebar.file_uploader(
    "CSVファイルをアップロード",
    type=['csv'],
    help="日付、店舗、売上、客数、イベントなどのデータを含むCSVファイル"
)

if uploaded_file is not None:
    # データを読み込み
    data = load_and_process_data(uploaded_file)
    
    if data is not None:
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
        display_cols = ['date', 'sales']
        if 'customers' in store_data.columns:
            display_cols.append('customers')
        if 'event' in store_data.columns:
            display_cols.append('event')
        
        st.dataframe(store_data.tail(10)[display_cols])
    
    # タブ2: モデル学習
    with tab2:
        st.header("Prophetモデル学習")
        
        # 学習する店舗を選択
        stores_to_train = st.multiselect(
            "学習する店舗を選択",
            data['store'].unique(),
            default=data['store'].unique()
        )
        
        # 高度な設定
        with st.expander("モデル設定（上級者向け）"):
            col1, col2 = st.columns(2)
            with col1:
                changepoint_prior_scale = st.slider(
                    "トレンドの柔軟性",
                    0.001, 0.5, 0.05,
                    help="大きいほどトレンドが柔軟に変化"
                )
                seasonality_prior_scale = st.slider(
                    "季節性の強さ",
                    0.01, 25.0, 10.0,
                    help="大きいほど季節性が強く反映"
                )
            with col2:
                seasonality_mode = st.selectbox(
                    "季節性モード",
                    ['multiplicative', 'additive'],
                    help="multiplicative: 売上に比例、additive: 一定額"
                )
                include_regressors = st.checkbox(
                    "イベント情報を使用",
                    value=True
                )
        
        if st.button("🚀 モデルを学習", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, store in enumerate(stores_to_train):
                status_text.text(f"学習中: {store}")
                store_data = data[data['store'] == store]
                
                if len(store_data) >= 30:
                    # Prophet用データの準備
                    prophet_df = prepare_prophet_data(store_data, include_regressors)
                    
                    # モデル学習
                    model_info = train_prophet_model(prophet_df, store)
                    st.session_state.models[store] = model_info
                    
                    # 結果を表示
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if model_info['mae'] > 0:
                            st.metric(f"{store} - MAE", f"¥{model_info['mae']:,.0f}")
                        else:
                            st.metric(f"{store} - MAE", "評価中...")
                    with col2:
                        if model_info['mape'] > 0:
                            st.metric(f"{store} - MAPE", f"{model_info['mape']:.1f}%")
                        else:
                            st.metric(f"{store} - MAPE", "評価中...")
                    with col3:
                        if model_info['rmse'] > 0:
                            st.metric(f"{store} - RMSE", f"¥{model_info['rmse']:,.0f}")
                        else:
                            st.metric(f"{store} - RMSE", "評価中...")
                    with col4:
                        if model_info['r2'] > 0:
                            st.metric(f"{store} - R²", f"{model_info['r2']:.3f}")
                        else:
                            st.metric(f"{store} - R²", "評価中...")
                    
                    # 曜日別精度を表示
                    if len(model_info['weekday_accuracy']) > 0:
                        with st.expander(f"{store} の曜日別予測精度"):
                            weekday_names = ['月', '火', '水', '木', '金', '土', '日']
                            accuracy_df = model_info['weekday_accuracy'].copy()
                            if len(accuracy_df) > 0:
                                accuracy_df.index = [weekday_names[i] for i in accuracy_df.index if i < len(weekday_names)]
                                accuracy_df.columns = ['実績平均', '予測平均']
                                accuracy_df['差異(%)'] = ((accuracy_df['予測平均'] / accuracy_df['実績平均'] - 1) * 100).round(1)
                                st.dataframe(accuracy_df)
                else:
                    st.warning(f"{store}のデータが不足しています（{len(store_data)}件）")
                
                progress_bar.progress((i + 1) / len(stores_to_train))
            
            status_text.text("✅ 全ての学習が完了しました！")
    
    # タブ3: 売上予測
    with tab3:
        st.header("売上予測")
        
        if not st.session_state.models:
            st.warning("⚠️ まずモデルを学習してください")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                forecast_store = st.selectbox(
                    "予測する店舗",
                    list(st.session_state.models.keys())
                )
            
            # 予測期間の選択（カレンダー式）
            store_data = data[data['store'] == forecast_store]
            first_date = store_data['date'].min().date()
            last_date = store_data['date'].max().date()
            
            with col2:
                start_date = st.date_input(
                    "予測開始日",
                    value=last_date + timedelta(days=1),
                    min_value=first_date,
                    max_value=last_date + timedelta(days=365),
                    help="過去の日付を選択すると、実績との比較が可能です"
                )
            
            with col3:
                default_end = min(start_date + timedelta(days=29), last_date + timedelta(days=60))
                end_date = st.date_input(
                    "予測終了日",
                    value=default_end,
                    min_value=start_date,
                    max_value=start_date + timedelta(days=365),
                    help="開始日から最大365日先まで選択可能"
                )
            
            # 予測日数を計算
            forecast_days = (end_date - start_date).days + 1
            
            # 過去データとの比較情報
            if start_date <= last_date:
                overlap_days = min(end_date, last_date) - start_date + timedelta(days=1)
                st.info(f"📊 予測期間: {start_date} ～ {end_date} ({forecast_days}日間)")
                if overlap_days.days > 0:
                    st.success(f"✅ {overlap_days.days}日分の実績データと比較可能です")
            else:
                st.info(f"🔮 予測期間: {start_date} ～ {end_date} ({forecast_days}日間)")
            
            if forecast_days > 365:
                st.error("予測期間は最大365日までです")
            elif forecast_days <= 0:
                st.error("予測期間を正しく設定してください")
            
            if st.button("📈 予測実行", type="primary") and 0 < forecast_days <= 365:
                # 予測を実行
                model_info = st.session_state.models[forecast_store]
                
                # Prophetの予測期間を計算（データの最初から予測終了日まで）
                days_from_start = (end_date - first_date).days + 1
                
                # 予測実行
                forecast_df, full_forecast = make_prophet_forecast(model_info, store_data, days_from_start)
                
                # 指定期間のデータのみを抽出
                forecast_df = forecast_df[
                    (forecast_df['date'].dt.date >= start_date) & 
                    (forecast_df['date'].dt.date <= end_date)
                ].copy()
                
                # 結果を保存
                st.session_state.forecast_results = {
                    'store': forecast_store,
                    'forecast': forecast_df,
                    'full_forecast': full_forecast,
                    'start_date': start_date,
                    'end_date': end_date
                }
                
                # 予測結果のサマリー
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("予測期間合計売上", f"¥{forecast_df['predicted_sales'].sum():,.0f}")
                with col2:
                    st.metric("日次平均売上", f"¥{forecast_df['predicted_sales'].mean():,.0f}")
                with col3:
                    st.metric("期間中の平均人時売上", f"¥{forecast_df['predicted_sales_per_hour'].mean():,.0f}")
                
                # 実績との比較（過去期間が含まれる場合）
                if start_date <= last_date:
                    actual_data = store_data[
                        (store_data['date'].dt.date >= start_date) & 
                        (store_data['date'].dt.date <= min(end_date, last_date))
                    ]
                    
                    if len(actual_data) > 0:
                        # 実績と予測の比較グラフ
                        st.subheader("実績と予測の比較")
                        
                        fig = go.Figure()
                        
                        # 実績データ
                        fig.add_trace(go.Scatter(
                            x=actual_data['date'],
                            y=actual_data['sales'],
                            mode='lines+markers',
                            name='実績',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # 予測データ（実績期間のみ）
                        forecast_overlap = forecast_df[forecast_df['date'].dt.date <= last_date]
                        if len(forecast_overlap) > 0:
                            fig.add_trace(go.Scatter(
                                x=forecast_overlap['date'],
                                y=forecast_overlap['predicted_sales'],
                                mode='lines+markers',
                                name='予測',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                            
                            # 予測区間
                            fig.add_trace(go.Scatter(
                                x=forecast_overlap['date'],
                                y=forecast_overlap['upper_bound'],
                                fill=None,
                                mode='lines',
                                line_color='rgba(255,0,0,0)',
                                showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=forecast_overlap['date'],
                                y=forecast_overlap['lower_bound'],
                                fill='tonexty',
                                mode='lines',
                                line_color='rgba(255,0,0,0)',
                                name='予測区間',
                                fillcolor='rgba(255,0,0,0.2)'
                            ))
                        
                        fig.update_layout(
                            xaxis_title="日付",
                            yaxis_title="売上（円）",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 精度評価
                        if len(forecast_overlap) > 0:
                            merged_data = actual_data.merge(
                                forecast_overlap[['date', 'predicted_sales']], 
                                on='date', 
                                how='inner'
                            )
                            
                            if len(merged_data) > 0:
                                mae = np.mean(np.abs(merged_data['sales'] - merged_data['predicted_sales']))
                                mape = np.mean(np.abs((merged_data['sales'] - merged_data['predicted_sales']) / merged_data['sales'])) * 100
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("比較期間のMAE", f"¥{mae:,.0f}")
                                with col2:
                                    st.metric("比較期間のMAPE", f"{mape:.1f}%")
                
                # Prophetのグラフ表示（全期間）
                st.subheader("売上予測グラフ（全期間）")
                fig = plot_plotly(model_info['model'], full_forecast)
                st.plotly_chart(fig, use_container_width=True)
                
                # コンポーネント分解
                st.subheader("予測の構成要素")
                fig_components = plot_components_plotly(model_info['model'], full_forecast)
                st.plotly_chart(fig_components, use_container_width=True)
                
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
                
                # 予測区間も表示
                display_df['予測区間'] = display_df.apply(
                    lambda row: f"¥{row['lower_bound']:,.0f} ~ ¥{row['upper_bound']:,.0f}",
                    axis=1
                )
                
                # 実績データがある場合は追加
                if start_date <= last_date:
                    actual_for_display = store_data[
                        (store_data['date'].dt.date >= start_date) & 
                        (store_data['date'].dt.date <= end_date)
                    ][['date', 'sales']].copy()
                    
                    if len(actual_for_display) > 0:
                        display_df = display_df.merge(
                            actual_for_display.rename(columns={'sales': 'actual_sales'}),
                            on='date',
                            how='left'
                        )
                        display_df['実績売上'] = display_df['actual_sales'].apply(
                            lambda x: f"¥{x:,.0f}" if pd.notna(x) else "-"
                        )
                        display_columns = ['日付', '店舗', '実績売上', '予測売上', '予測客数', '予測労働時間', '予測人時', '予測区間']
                    else:
                        display_columns = ['日付', '店舗', '予測売上', '予測客数', '予測労働時間', '予測人時', '予測区間']
                else:
                    display_columns = ['日付', '店舗', '予測売上', '予測客数', '予測労働時間', '予測人時', '予測区間']
                
                display_df = display_df[display_columns]
                
                # データフレームを表示（スクロール可能）
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
                
                # 予測データのダウンロード
                st.subheader("データダウンロード")
                
                export_df = forecast_df.copy()
                weekday_names_export = ['月', '火', '水', '木', '金', '土', '日']
                export_df['曜日'] = export_df['date'].dt.dayofweek.apply(lambda x: weekday_names_export[x])
                export_df['日付'] = export_df['date'].dt.strftime('%Y-%m-%d')
                export_df['店舗'] = forecast_store
                export_df = export_df.rename(columns={
                    'predicted_sales': '予測売上',
                    'predicted_customers': '予測客数',
                    'predicted_labor_hours': '予測労働時間',
                    'predicted_sales_per_hour': '予測人時',
                    'lower_bound': '予測下限',
                    'upper_bound': '予測上限'
                })
                
                # 実績データも含める（あれば）
                if start_date <= last_date:
                    actual_export = store_data[
                        (store_data['date'].dt.date >= start_date) & 
                        (store_data['date'].dt.date <= end_date)
                    ][['date', 'sales']].copy()
                    
                    if len(actual_export) > 0:
                        export_df = export_df.merge(
                            actual_export.rename(columns={'sales': '実績売上'}),
                            on='date',
                            how='left'
                        )
                        export_df = export_df[['日付', '曜日', '店舗', '実績売上', '予測売上', '予測客数', '予測労働時間', '予測人時', '予測下限', '予測上限']]
                    else:
                        export_df = export_df[['日付', '曜日', '店舗', '予測売上', '予測客数', '予測労働時間', '予測人時', '予測下限', '予測上限']]
                else:
                    export_df = export_df[['日付', '曜日', '店舗', '予測売上', '予測客数', '予測労働時間', '予測人時', '予測下限', '予測上限']]
                
                # ダウンロードオプション
                col1, col2 = st.columns(2)
                
                with col1:
                    # UTF-8 BOM付き（Excel対応）
                    csv_utf8 = export_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 ダウンロード (Excel/Mac対応)",
                        data=csv_utf8,
                        file_name=f'{forecast_store}_prophet_forecast_{datetime.now().strftime("%Y%m%d")}_utf8.csv',
                        mime='text/csv',
                        help="UTF-8 BOM付き形式。Excel、Numbers、Google Sheetsで文字化けせずに開けます"
                    )
                
                with col2:
                    # Shift-JIS（Windows Excel用）
                    try:
                        csv_sjis = export_df.to_csv(index=False, encoding='shift-jis')
                        st.download_button(
                            label="📥 ダウンロード (Windows専用)",
                            data=csv_sjis,
                            file_name=f'{forecast_store}_prophet_forecast_{datetime.now().strftime("%Y%m%d")}_sjis.csv',
                            mime='text/csv',
                            help="Shift-JIS形式。古いWindows Excelで開く場合に使用"
                        )
                    except:
                        st.info("Shift-JIS変換エラー。UTF-8版をご利用ください。")
    
    # タブ4: 分析レポート
    with tab4:
        st.header("分析レポート")
        
        if st.session_state.models:
            # 全店舗の予測精度比較
            st.subheader("店舗別予測精度")
            
            comparison_data = []
            for store, info in st.session_state.models.items():
                data_row = {'店舗': store}
                if info['mae'] > 0:
                    data_row['MAE'] = info['mae']
                    data_row['MAPE'] = info['mape']
                    data_row['RMSE'] = info['rmse']
                    data_row['R²'] = info['r2']
                else:
                    data_row['MAE'] = None
                    data_row['MAPE'] = None
                    data_row['RMSE'] = None
                    data_row['R²'] = None
                comparison_data.append(data_row)
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # NaNを除外してグラフ作成
                valid_df = comparison_df.dropna(subset=['MAPE'])
                
                if len(valid_df) > 0:
                    # MAPEでソートして棒グラフ
                    valid_df = valid_df.sort_values('MAPE')
                    
                    fig = px.bar(
                        valid_df,
                        x='店舗',
                        y='MAPE',
                        title="店舗別予測精度（MAPE: 低いほど良い）",
                        color='MAPE',
                        color_continuous_scale='RdYlGn_r'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 詳細な評価指標テーブル
                    st.subheader("評価指標詳細")
                    display_df = valid_df.copy()
                    display_df['MAE'] = display_df['MAE'].apply(lambda x: f"¥{x:,.0f}" if pd.notna(x) else "-")
                    display_df['MAPE'] = display_df['MAPE'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
                    display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"¥{x:,.0f}" if pd.notna(x) else "-")
                    display_df['R²'] = display_df['R²'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
                    st.dataframe(display_df)
                else:
                    st.info("評価指標を計算中です。十分なテストデータがない可能性があります。")
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
            
            # Prophetモデルの特徴
            st.subheader("Prophetモデルの特徴")
            st.markdown("""
            **Prophetの利点:**
            - 📈 トレンドを自動的に検出・予測
            - 📅 週次・年次の季節性を自動モデル化
            - 🎌 祝日効果を考慮可能
            - 📊 予測区間（信頼区間）を提供
            - 🔧 欠損値や外れ値に対してロバスト
            
            **適用シーン:**
            - 長期的なトレンドがあるデータ
            - 明確な季節性パターンがあるデータ
            - 祝日やイベントの影響が大きいビジネス
            """)

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
    
    ※ Prophetは時系列データに特化しているため、日付と売上のデータがあれば基本的な予測が可能です。
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