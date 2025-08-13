import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Rectangle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
# Makine öğrenimi modelleri için
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# Verileri görselleştirmek için
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
# Uyarıları kapatmak için
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
from datetime import datetime, timedelta
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
import streamlit as st
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# 1. ORIGINAL FUNCTIONS (Preserved)
def get_stock_data(symbols, start_date, end_date):
    df = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
    final_df = pd.DataFrame()
    for symbol in symbols:
        temp = df[symbol].copy()
        temp['stock_symbol'] = symbol.replace(".IS", "")
        temp['date'] = temp.index
        temp = temp[['stock_symbol', 'date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        temp.columns = ['stock_symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        final_df = pd.concat([final_df, temp])
    return final_df.reset_index(drop=True)


def add_technical_indicators(df):
    df["RSI_14"] = ta.rsi(df["close"], length=14)
    df["Daily_Return"] = df["close"].pct_change()
    bbands = ta.bbands(df["close"], length=20, std=2)
    df["BBL"] = bbands.iloc[:, 0]
    df["BBM"] = bbands.iloc[:, 1]
    df["BBU"] = bbands.iloc[:, 2]
    df["BB_width"] = df["BBU"] - df["BBL"]
    df["CCI"] = ta.cci(df["high"], df["low"], df["close"], length=20)
    df["OBV"] = ta.obv(df["close"], df["volume"])
    df["daily_volatility"] = df["high"] - df["low"]
    df["rolling_std_14"] = df["close"].rolling(window=14).std()
    df["momentum_10"] = df["close"] - df["close"].shift(10)
    df["Rolling_Vol"] = df["Daily_Return"].rolling(7).std()
    df["Price_Vol_Ratio"] = df["close"] / df["Rolling_Vol"]
    return df.dropna()


def prepare_data(df, window_size=45, forecast_horizon=1):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    def create_sequences(data, window_size, forecast_horizon):
        X, y = [], []
        for i in range(window_size, len(data) - forecast_horizon):
            X.append(data[i - window_size:i])
            y.append(data[i + forecast_horizon][3])  # 3 is the index of 'close' price
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, window_size, forecast_horizon)
    return train_test_split(X, y, test_size=0.2, shuffle=False), scaler


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def predict_next_day(model, last_sequence, scaler):
    next_day_scaled = model.predict(last_sequence)
    close_scaler = MinMaxScaler()
    close_scaler.min_, close_scaler.scale_ = scaler.min_[3], scaler.scale_[3]
    return close_scaler.inverse_transform(next_day_scaled)[0][0]


def evaluate_model(y_true, y_pred):
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    return metrics


def plot_training_history(history):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history.history['loss'], label='Eğitim Kaybı', color='blue', marker='o', markersize=4)
    ax.plot(history.history['val_loss'], label='Test Kaybı', color='red', marker='x', markersize=4)
    ax.set_title('Model Kaybı: Overfitting Kontrolü', fontsize=14, pad=20)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    y_max = max(max(history.history['loss']), max(history.history['val_loss'])) * 1.1
    ax.set_ylim(0, y_max)
    ax.set_xticks(range(0, len(history.history['loss']), max(1, len(history.history['loss']) // 10)))
    return fig


# 2. NEW GRAPH FUNCTIONS WITH CONSISTENT SIZING
def plot_daily_returns(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df['Daily_Return'], bins=50, color='green', alpha=0.7)
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequency')
    ax.set_title('Günlük Getiri Dağılımı')
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig


def plot_trading_volume(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df['date'], df['volume'], color='orange', alpha=0.7)
    ax.set_xlabel('Tarih')
    ax.set_ylabel('Hacim')
    ax.set_title('İşlem Hacmi Zaman Serisi')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.autofmt_xdate()
    return fig


def plot_correlation_matrix(df):
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.drop(['stock_symbol', 'date'], axis=1, errors='ignore').corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, center=0)
    ax.set_title('Korelasyon Matrisi')
    return fig


def plot_volume_vs_return(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(df['volume'], df['Daily_Return'], alpha=0.5, color='purple')
    ax.set_xlabel('Hacim')
    ax.set_ylabel('Günlük Getiri')
    ax.set_title('Hacim vs Getiri İlişkisi')
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig


def plot_rsi(df):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df["RSI_14"], label="RSI_14", color='purple')
    ax.axhline(70, color='red', linestyle='--', label='Aşırı Alım')
    ax.axhline(30, color='green', linestyle='--', label='Aşırı Satım')
    ax.set_title("RSI 14 Göstergesi")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("RSI Değeri")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.autofmt_xdate()
    return fig


def plot_bollinger_bands(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["date"], df["close"], label="Kapanış", color="black")
    ax.plot(df["date"], df["BBU"], label="Üst Bant", linestyle='--', color='blue')
    ax.plot(df["date"], df["BBM"], label="Orta Bant", linestyle='--', color='green')
    ax.plot(df["date"], df["BBL"], label="Alt Bant", linestyle='--', color='red')
    ax.fill_between(df["date"], df["BBL"], df["BBU"], color='lightgray', alpha=0.3)
    ax.set_title("Bollinger Bantları")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Fiyat")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.autofmt_xdate()
    return fig


def plot_obv(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title("Fiyat ve OBV Göstergesi")
    ax1.plot(df["date"], df["close"], label="Kapanış Fiyatı", color='blue')
    ax1.set_ylabel("Fiyat")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["OBV"], label="OBV", color='orange')
    ax2.set_ylabel("OBV")
    ax2.legend(loc="upper right")

    fig.autofmt_xdate()
    return fig


def plot_momentum(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(df["date"], df["momentum_10"], color='teal')
    ax1.set_title("10 Günlük Momentum")
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2.plot(df["date"], df["Daily_Return"], color='brown')
    ax2.set_title("Günlük Getiriler")
    ax2.grid(True, linestyle='--', alpha=0.3)

    fig.autofmt_xdate()
    return fig


# 3. PREDICTION VISUALIZATION FUNCTIONS WITH TOOLTIPS
def create_prediction_summary(bugünkü_fiyat, yarınki_tahmin):
    fark = yarınki_tahmin - bugünkü_fiyat
    change_percent = (fark / bugünkü_fiyat) * 100
    yön = "YÜKSELİŞ" if fark > 0 else "DÜŞÜŞ"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#f5f5f5')
    ax.grid(color='white', linestyle='-', linewidth=1)

    text_box = AnchoredText(f"""
    BUGÜN: {bugünkü_fiyat:.2f} ₺
    YARIN: {yarınki_tahmin:.2f} ₺
    FARK: {abs(fark):.2f} ₺ ({change_percent:.2f}%)
    YÖN: {yön}
    """, loc='center', frameon=True, prop=dict(size=12))
    ax.add_artist(text_box)

    arrow_color = 'red' if fark < 0 else 'green'
    arrow_symbol = '↓' if fark < 0 else '↑'
    plt.text(0.5, 0.7, arrow_symbol, fontsize=60,
             color=arrow_color, ha='center', va='center', alpha=0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title('FİYAT TAHMİN PANOSU', pad=20)
    return fig, f"Bugünkü fiyat: {bugünkü_fiyat:.2f} ₺, Yarın tahmini: {yarınki_tahmin:.2f} ₺ ({change_percent:.2f}%)"


def create_signal_chart(bugünkü_fiyat, yarınki_tahmin):
    fark = yarınki_tahmin - bugünkü_fiyat
    yüzde_değişim = (fark / bugünkü_fiyat) * 100

    if yarınki_tahmin > bugünkü_fiyat * 1.01:  # %1'den fazla artış
        signal = "BUY"
        renk = "green"
        explanation = "Fiyatın yükseleceği öngörülüyor - AL sinyali"
    elif yarınki_tahmin < bugünkü_fiyat * 0.99:  # %1'den fazla düşüş
        signal = "SELL"
        renk = "red"
        explanation = "Fiyatın düşeceği öngörülüyor - SAT sinyali"
    else:
        signal = "HOLD"  # Nötr
        renk = "gray"
        explanation = "Belirgin bir trend öngörülmüyor - BEKLE sinyali"

    fig, ax = plt.subplots(figsize=(12, 6))

    # Fiyat çubukları
    ax.bar(["Bugün", "Yarın"], [bugünkü_fiyat, yarınki_tahmin],
           color=["blue", renk], alpha=0.6)

    # Bağlantı çizgisi
    ax.plot(["Bugün", "Yarın"], [bugünkü_fiyat, yarınki_tahmin],
            color=renk, linestyle="--", marker="o")

    # Bilgi metni
    ax.text(0.5, (bugünkü_fiyat + yarınki_tahmin) / 2,
            f"{yüzde_değişim:.2f}%",
            ha='center', va='center', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))

    # Sinyal kutusu
    signal_box = Rectangle((0, max(bugünkü_fiyat, yarınki_tahmin) * 1.1),
                           2, max(bugünkü_fiyat, yarınki_tahmin) * 0.1,
                           facecolor=renk, alpha=0.3)
    ax.add_patch(signal_box)
    ax.text(1, max(bugünkü_fiyat, yarınki_tahmin) * 1.15,
            signal, ha='center', va='center',
            fontsize=20, weight='bold', color=renk)

    # Grafik ayarları
    ax.set_title(f"Fiyat Tahmini: {signal} Sinyali - {explanation}", pad=20)
    ax.set_ylabel("Fiyat (₺)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(min(bugünkü_fiyat, yarınki_tahmin) * 0.95,
                max(bugünkü_fiyat, yarınki_tahmin) * 1.2)

    return fig, signal, yüzde_değişim, explanation


# 4. STREAMLIT MAIN APPLICATION WITH IMPROVED LAYOUT
def main():
    st.set_page_config(layout="wide", page_title="Hisse Analiz ve Tahmin Paneli", page_icon="📈")
    st.title("📊 Hisse Senedi Analiz ve Tahmin Paneli")

    with st.sidebar:
        st.header("🔧 Parametreler")
        symbol = st.text_input("Hisse Kodu", "TUPRS.IS")
        start_date = st.date_input("Başlangıç Tarihi", datetime(2021, 1, 1))
        end_date = st.date_input("Bitiş Tarihi", datetime.now())
        run_analysis = st.button("Analiz ve Tahmin Yap")

    if run_analysis:
        try:
            with st.spinner("Veri yükleniyor ve analiz ediliyor..."):
                # 1. Data Loading and Initial Analysis
                df = get_stock_data([symbol], start_date, end_date)
                df = add_technical_indicators(df)

                # Get today's price for prediction visualizations
                bugünkü_fiyat = df.iloc[-1]['close']

                # 2. Technical Analysis Graphs
                st.header("📈 Teknik Analiz Grafikleri")

                tab1, tab2, tab3 = st.tabs(["Temel Göstergeler", "Teknik Göstergeler", "Korelasyon Analizi"])

                with tab1:
                    with st.expander(
                            "📊 Günlük Getiri Dağılımı - Hisse getirilerinin istatistiksel dağılımını gösterir"):
                        st.pyplot(plot_daily_returns(df))
                        st.markdown("""
                        **Ne İşe Yarar?**  
                        Hisse senedinin günlük getirilerinin nasıl dağıldığını gösterir. Normal dağılıma yakın olması beklenir.
                        """)

                    with st.expander("📈 İşlem Hacmi Zaman Serisi - Günlük işlem hacmini gösterir"):
                        st.pyplot(plot_trading_volume(df))
                        st.markdown("""
                        **Ne İşe Yarar?**  
                        Hisse senedinin günlük işlem hacmini gösterir. Yüksek hacimler genellikle önemli fiyat hareketlerine eşlik eder.
                        """)

                    with st.expander("🔄 Hacim vs Getiri İlişkisi - Hacim ve getiri arasındaki ilişkiyi gösterir"):
                        st.pyplot(plot_volume_vs_return(df))
                        st.markdown("""
                        **Ne İşe Yarar?**  
                        Hacim ve getiri arasındaki ilişkiyi gösterir. Yüksek hacimlerin hangi yönde getiri sağladığını analiz etmeye yardımcı olur.
                        """)

                with tab2:
                    with st.expander("📉 RSI 14 Göstergesi - Aşırı alım/satım bölgelerini gösterir"):
                        st.pyplot(plot_rsi(df))
                        st.markdown("""
                        **Ne İşe Yarar?**  
                        RSI (Relative Strength Index), hissenin aşırı alım (70 üstü) veya aşırı satım (30 altı) bölgelerinde olup olmadığını gösterir.
                        """)

                    with st.expander("📊 Bollinger Bantları - Fiyat volatilitesini gösterir"):
                        st.pyplot(plot_bollinger_bands(df))
                        st.markdown("""
                        **Ne İşe Yarar?**  
                        Bollinger Bantları, fiyatın volatilitesini ve potansiyel destek/direnç seviyelerini gösterir. Fiyat genellikle bantlar arasında hareket eder.
                        """)

                    with st.expander("💹 Fiyat ve OBV Göstergesi - Hacim akışını gösterir"):
                        st.pyplot(plot_obv(df))
                        st.markdown("""
                        **Ne İşe Yarar?**  
                        OBV (On Balance Volume), hacim akışını gösterir. Fiyatla uyumlu hareket ediyorsa trendin güçlü olduğuna işaret eder.
                        """)

                    with st.expander("🚀 Momentum ve Getiriler - Fiyat momentumunu gösterir"):
                        st.pyplot(plot_momentum(df))
                        st.markdown("""
                        **Ne İşe Yarar?**  
                        Üst grafik 10 günlük momentumu, alt grafik günlük getirileri gösterir. Momentum, trendin gücünü anlamaya yardımcı olur.
                        """)

                with tab3:
                    with st.expander("🔗 Korelasyon Matrisi - Göstergeler arası ilişkileri gösterir"):
                        st.pyplot(plot_correlation_matrix(df))
                        st.markdown("""
                        **Ne İşe Yarar?**  
                        Tüm teknik göstergelerin birbirleriyle olan ilişkisini gösterir. Yüksek korelasyonlu göstergeler benzer bilgi sağlıyor olabilir.
                        """)

                # 3. Model Training and Prediction
                st.header("🤖 Tahmin Modeli Sonuçları")

                with st.spinner("Model eğitiliyor..."):
                    df_model = df.drop(['stock_symbol', 'date'], axis=1)
                    (X_train, X_test, y_train, y_test), scaler = prepare_data(df_model)

                    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
                    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True,
                                               min_delta=0.0001)
                    history = model.fit(X_train, y_train, epochs=50,
                                        validation_data=(X_test, y_test),
                                        callbacks=[early_stop], verbose=0)

                    # Model Results
                    y_pred = model.predict(X_test)
                    metrics = evaluate_model(y_test, y_pred)

                    # Prediction for next day
                    last_sequence = df_model[-45:].values  # window_size = 45
                    last_sequence = scaler.transform(last_sequence)
                    last_sequence = np.expand_dims(last_sequence, axis=0)
                    yarınki_tahmin = predict_next_day(model, last_sequence, scaler)

                    # Display results in columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Model Performansı")
                        st.dataframe(pd.DataFrame.from_dict(metrics, orient='index', columns=['Değer']))
                        st.markdown("""
                        **Metrik Açıklamaları:**  
                        - MAE: Ortalama Mutlak Hata  
                        - MSE: Ortalama Kare Hata  
                        - RMSE: Kök Ortalama Kare Hata  
                        - R2: Açıklanan Varyans Oranı  
                        - MAPE: Ortalama Mutlak Yüzde Hata
                        """)

                    with col2:
                        st.subheader("1 Gün Sonrası Tahmin")
                        st.metric(label="Tahmini Kapanış Fiyatı", value=f"{yarınki_tahmin:.2f} ₺",
                                  delta=f"{(yarınki_tahmin - bugünkü_fiyat):.2f} ₺ ({(yarınki_tahmin - bugünkü_fiyat) / bugünkü_fiyat * 100:.2f}%)")
                        st.markdown(f"""
                        **Bugünkü Fiyat:** {bugünkü_fiyat:.2f} ₺  
                        **Tahmini Fiyat:** {yarınki_tahmin:.2f} ₺  
                        **Fark:** {yarınki_tahmin - bugünkü_fiyat:.2f} ₺  
                        **Yön:** {'Yükseliş' if yarınki_tahmin > bugünkü_fiyat else 'Düşüş'}
                        """)

                    # New prediction visualizations
                    st.subheader("Tahmin Görselleştirmeleri")

                    col3, col4 = st.columns(2)

                    with col3:
                        with st.expander("📌 Fiyat Tahmin Panosu - Özet Bilgiler"):
                            summary_fig, summary_text = create_prediction_summary(bugünkü_fiyat, yarınki_tahmin)
                            st.pyplot(summary_fig)
                            st.markdown(f"**Açıklama:** {summary_text}")

                    with col4:
                        with st.expander("🚦 Al/Sat Sinyali - Trading Karar Destek"):
                            signal_fig, signal, yüzde_değişim, explanation = create_signal_chart(bugünkü_fiyat,
                                                                                                 yarınki_tahmin)
                            st.pyplot(signal_fig)
                            st.markdown(f"""
                            **Sinyal Türü:** {signal}  
                            **Değişim Oranı:** {yüzde_değişim:.2f}%  
                            **Açıklama:** {explanation}
                            """)

                    # Display trading signal
                    if signal == "BUY":
                        st.success(f"**📢 AL SİNYALİ ({yüzde_değişim:.2f}%)**: Fiyatın yükseleceği öngörülüyor")
                    elif signal == "SELL":
                        st.error(f"**📢 SAT SİNYALİ ({yüzde_değişim:.2f}%)**: Fiyatın düşeceği öngörülüyor")
                    else:
                        st.info(f"**📢 BEKLE SİNYALİ ({yüzde_değişim:.2f}%)**: Belirgin bir trend öngörülmüyor")

                    # Training History
                    with st.expander("📈 Model Eğitim Süreci - Overfitting Kontrolü"):
                        st.pyplot(plot_training_history(history))
                        st.markdown("""
                        **Ne İşe Yarar?**  
                        Modelin eğitim ve validasyon kaybını gösterir. İki çizgi birbirine yakın seyrediyorsa model iyi genelleme yapıyor demektir.
                        """)

        except Exception as e:
            st.error(f"Hata oluştu: {str(e)}")
            st.stop()


if __name__ == "__main__":
    main()


