import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
 
# ============================================
# 위대리 → 고평가/중립/저평가 단계로 구분
# ============================================
   
# ============================================
# 1) 일간 → 금요일 기준 주간 데이터 변환
# ============================================

@st.cache_data(ttl=3600) # 1시간 캐시
def get_weekly_data_from_daily(ticker, start_date):
    """
    FinanceDataReader 일간 데이터를 가져와서
    금요일 기준으로 주간 종가를 만들기
    (금요일 휴장 시 목요일 데이터 자동 사용)
    """
    df_daily = fdr.DataReader(ticker, start=start_date)

    if 'Close' not in df_daily.columns:
        raise ValueError(f"{ticker} 데이터에 'Close' 컬럼이 없습니다.")

    # 금요일 기준 주간 종가
    df_weekly = df_daily['Close'].resample('W-FRI').last().dropna()
    return df_weekly


# ============================================
# 2) 성장(GROWTH) 추세선 계산
# ============================================

def calculate_growth_trend(prices, dates):
    """
    로그 가격 기반 추세선 계산 (최근 260주)
    """
    # 최근 260주(약 5년) 데이터만 사용
    if len(prices) > 260:
        prices = prices[-260:]
        dates = dates[-260:]

    dates_ordinal = np.array([d.toordinal() for d in dates])
    prices_array = np.array(prices)

    valid_mask = prices_array > 0
    if not np.any(valid_mask):
        return 0

    dates_ordinal = dates_ordinal[valid_mask]
    prices_array = prices_array[valid_mask]

    log_prices = np.log(prices_array)
    slope, intercept = np.polyfit(dates_ordinal, log_prices, 1)

    return np.exp(slope * dates_ordinal[-1] + intercept)


# ============================================
# 3) 메인 리밸런싱 로직 (Streamlit UI)
# ============================================

def run_rebalancing_logic(target_ticker, my_shares, my_cash, analysis_date):
    """
    메인 로직을 실행하고 결과를 딕셔너리로 반환
    """
    with st.spinner("📡 데이터를 분석 중입니다..."):
        fetch_start_date = "2010-01-01"

        # 주간 데이터 변환
        qqq = get_weekly_data_from_daily("QQQ", fetch_start_date)
        df_target = get_weekly_data_from_daily(target_ticker, fetch_start_date)

        # 사용자 지정 분석 기준일 이전 데이터만 사용
        analysis_date_dt = pd.to_datetime(analysis_date)
        qqq = qqq[qqq.index <= analysis_date_dt]
        df_target = df_target[df_target.index <= analysis_date_dt]

        # 2010-02-12 이후만 사용
        start_filter = "2010-02-12"
        qqq = qqq[qqq.index >= start_filter]
        df_target = df_target[df_target.index >= start_filter]

        if len(qqq) < 2 or len(df_target) < 2:
            st.error(f"'{analysis_date.strftime('%Y-%m-%d')}'까지의 데이터가 부족하여 분석할 수 없습니다. 더 이전 날짜를 선택해주세요.")
            st.stop()

        # 기준 날짜 (마지막 금요일)
        last_date = qqq.index[-1].strftime("%Y-%m-%d")

        # 추세선 가격
        trend_price = calculate_growth_trend(qqq.values, qqq.index)

        # --- 차트 데이터 생성 (최근 260주) ---
        qqq_plot = qqq.copy()
        if len(qqq_plot) > 260:
            qqq_plot = qqq_plot[-260:]
        
        dates_ordinal = np.array([d.toordinal() for d in qqq_plot.index])
        log_prices = np.log(qqq_plot.values)
        slope, intercept = np.polyfit(dates_ordinal, log_prices, 1)
        trend_values = np.exp(slope * dates_ordinal + intercept)
        
        df_chart = pd.DataFrame({"QQQ": qqq_plot.values, "Trend": trend_values}, index=qqq_plot.index)

        curr_qqq = qqq.iloc[-1]
        curr_target = df_target.iloc[-1]
        prev_target = df_target.iloc[-2] if len(df_target) > 1 else curr_target

        # 이격도
        gap = (curr_qqq - trend_price) / trend_price

        # 상태 판정
        if gap > 0.05:
            status = "🔴 고평가"; sell_rate, buy_rate = 1.00, 0.67
        elif gap < -0.06:
            status = "🔵 저평가"; sell_rate, buy_rate = 0.75, 1.50
        else:
            status = "🟢 중립"; sell_rate, buy_rate = 0.67, 0.67

        # 변동 계산
        price_change = curr_target - prev_target
        pnl = price_change * my_shares
        change_pct = (price_change / prev_target) * 100 if prev_target != 0 else 0

        # 매매 계산
        warning = ""
        if pnl > 0:
            # 매도
            action = "매도"
            trade_amount = pnl * sell_rate
            trade_shares = round(trade_amount / curr_target)
        elif pnl < 0:
            # 매수
            action = "매수"
            need = abs(pnl) * buy_rate
            if need > my_cash:
                need = my_cash
                warning = "(보유 현금으로 조정됨)"
            trade_shares = round(need / curr_target)

            # 초과 확인
            if trade_shares * curr_target > my_cash:
                trade_shares = int(my_cash // curr_target)
        else:
            action = "없음"
            trade_shares = 0

    # 결과를 딕셔너리로 반환
    return {
        "last_date": last_date,
        "curr_qqq": curr_qqq,
        "trend_price": trend_price,
        "curr_target": curr_target,
        "gap": gap,
        "status": status,
        "price_change": price_change,
        "change_pct": change_pct,
        "pnl": pnl,
        "action": action,
        "trade_shares": trade_shares,
        "warning": warning,
        "df_chart": df_chart
    }

# ============================================
# Streamlit UI 구성
# ============================================

st.title("📅 스마트 밸류 리밸런싱")

st.markdown("---")

# 분석 기간 설정
st.subheader("🗓️ 분석 기간 설정")
col_date1, col_date2 = st.columns(2)

with col_date1:
    start_date_input = st.date_input("👉 투자시작일", value=datetime(2024, 1, 1))

with col_date2:
    # 가장 최근 금요일을 기본값으로 설정
    today = datetime.today()
    offset = (today.weekday() - 4) % 7 # Friday is 4
    last_friday = today - timedelta(days=offset)
    end_date_input = st.date_input("👉 투자종료일 (분석 기준일)", value=last_friday)

st.markdown("---")

# 자산 입력
st.subheader("💰 내 자산 입력")

ticker_options = ['TQQQ', 'SOXL', 'TECL', 'UPRO', 'USD', 'FNGU', 'TSLL', '직접 입력']
selected_ticker = st.selectbox("👉 종목 선택", ticker_options)

if selected_ticker == '직접 입력':
    target_ticker = st.text_input("👉 종목 코드 입력", value="").upper()
else:
    target_ticker = selected_ticker

initial_asset = st.number_input("👉 초기자산 (원금, USD)", min_value=0.0, value=20000.0, step=1000.0)

col1, col2 = st.columns(2)
with col1:
    my_shares = st.number_input(f"👉 보유 {target_ticker} 수량 (주)", min_value=0.0, value=100.0, step=1.0)
with col2:
    my_cash = st.number_input("👉 매수 가능 현금 (USD)", min_value=0.0, value=10000.0, step=100.0)

st.markdown("---")

# 분석 시작 버튼
if st.button("🚀 리밸런싱 분석 시작!"):
    # 입력값 유효성 검사
    if not target_ticker:
        st.warning("종목 코드를 입력하거나 선택해주세요.")
        st.stop()
    
    if start_date_input > end_date_input:
        st.error("투자시작일은 투자종료일보다 이전이어야 합니다.")
        st.stop()

    results = run_rebalancing_logic(target_ticker, my_shares, my_cash, end_date_input)
    
    st.subheader("📈 총 자산 현황")
    current_stock_value = my_shares * results['curr_target']
    current_total_asset = current_stock_value + my_cash
    total_pnl = current_total_asset - initial_asset
    total_return_rate = (total_pnl / initial_asset) * 100 if initial_asset > 0 else 0.0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="현재 총자산", value=f"${current_total_asset:,.2f}", delta=f"${total_pnl:,.2f} (총 손익)")
    with col2:
        st.metric(label="초기 자산", value=f"${initial_asset:,.2f}")
    with col3:
        st.metric(label="총 수익률", value=f"{total_return_rate:.2f}%")

    st.markdown("---")

    st.header(f"📅 기준 날짜: {results['last_date']} (Weekly Close)")
    st.markdown("---")

    # [1] 시장 진단
    st.subheader("[1] 시장 진단")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="QQQ 현재가", value=f"${results['curr_qqq']:.2f}")
        st.metric(label=f"{target_ticker} 현재가", value=f"${results['curr_target']:.2f}")
    with col2:
        st.metric(label="QQQ 적정가 (GROWTH)", value=f"${results['trend_price']:.2f}")
        st.metric(label="이격도", value=f"{results['gap']*100:.2f}%", delta=results['status'])

    st.markdown("---")
    
    st.subheader("📊 QQQ 주가 및 추세선 (최근 5년)")
    st.line_chart(results['df_chart'], color=["#0000FF", "#FF0000"])
    st.markdown("---")

    # [2] 변동 분석
    st.subheader("[2] 변동 분석")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"주간 {target_ticker} 변동", value=f"${results['price_change']:.2f}", delta=f"{results['change_pct']:.2f}%")
    with col2:
        st.metric(label="내 평가금 변동", value=f"${results['pnl']:,.2f} USD")

    st.markdown("---")

    # [3] 이번 주 매매 제안
    st.subheader("📢 [이번 주 매매 제안]")
    if results['trade_shares'] == 0:
        st.info("👉 매매 없음 (변동폭 미미)")
    else:
        action_color = "blue" if results['action'] == "매수" else "red"
        st.markdown(f"### <font color='{action_color}'>👉 {target_ticker} {int(results['trade_shares'])}주 {results['action']}</font>", unsafe_allow_html=True)
        
        trade_value = results['trade_shares'] * results['curr_target']
        st.markdown(f"#### 예상 금액: `${trade_value:,.2f}`")
        
        if results['warning']:
            st.warning(f"⚠️ {results['warning']}")

    st.markdown("---")
