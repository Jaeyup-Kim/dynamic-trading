import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from collections import namedtuple
import numpy as np
import FinanceDataReader as fdr
import io
import json

# 파일 경로 정의
CONFIG_FILE = 'config.json'

### ---------------------------------------
# ✅ 설정 및 파라미터 저장/불러오기 함수
### ---------------------------------------
def load_config():
    """사용자 이름과 같은 전역 설정을 불러옵니다."""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # 파일이 없거나 형식이 잘못된 경우 초기값 반환
        return {
            "user_names": [f"사용자{i+1}" for i in range(6)]
        }

def save_config(config):
    """사용자 이름과 같은 전역 설정을 저장합니다."""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def get_params_file(user):
    """사용자 이름에 따라 파라미터 파일 경로를 반환합니다."""
    return f'params_{user}.json'

def load_params(user):
    """특정 사용자의 파라미터를 불러옵니다."""
    file_path = get_params_file(user)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # 파일이 없거나 형식이 잘못된 경우 초기값 반환
        return {
            "style_option": "Default",
            "target_ticker": "SOXL",
            "first_amt": 24000,
            "start_date": (datetime.today() - timedelta(days=21)).strftime('%Y-%m-%d'),
            "end_date": datetime.today().strftime('%Y-%m-%d')
        }

def save_params(params, user):
    """특정 사용자의 파라미터를 저장합니다."""
    file_path = get_params_file(user)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4, ensure_ascii=False)

### ---------------------------------------
# ✅ RSI 계산 함수
### ---------------------------------------
def calculate_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---------------------------------------
# ✅ 주간 RSI용 주차 계산 함수
# ---------------------------------------
def get_week_num(date):
    return int(date.strftime("%Y%U"))

# ---------------------------------------
# ✅ 주요 파라미터 (전략 설정값)
# ---------------------------------------

# 투자금 갱신 설정
INVT_RENWL_CYLE = 10

# 주문 정보 구조 정의
Order = namedtuple('Order', ['side', 'type', 'price', 'quantity'])

# ---------- 유틸 함수들 ----------
def get_weeknum_google_style(date):
    jan1 = pd.Timestamp(year=date.year, month=1, day=1).tz_localize(None)
    date = pd.Timestamp(date).tz_localize(None)
    weekday_jan1 = jan1.weekday()
    delta_days = (date - jan1).days
    return ((delta_days + weekday_jan1) // 7) + 1

def get_last_trading_day_each_week(data):
    data = data.copy()
    data['week'] = data.index.to_series().apply(get_weeknum_google_style)
    data['year'] = data.index.to_series().dt.year
    data['weekday'] = data.index.to_series().dt.weekday
    last_day = data.groupby(['year', 'week'])[['weekday']].idxmax()
    return data.loc[last_day['weekday']]

def calculate_rsi_rolling(data, period=14):
    data = data.copy()
    data['delta'] = data['Close'].diff()
    data['gain'] = data['delta'].where(data['delta'] > 0, 0.0)
    data['loss'] = -data['delta'].where(data['delta'] < 0, 0.0)
    data['avg_gain'] = data['gain'].rolling(window=period).mean()
    data['avg_loss'] = data['loss'].rolling(window=period).mean()
    data['RS'] = (data['avg_gain'] / data['avg_loss']).round(3)
    data['RSI'] = ((data['RS'] / (1 + data['RS'])) * 100).round(2)
    return data

def assign_mode_v2(rsi_series):
    mode_list = []
    for i in range(len(rsi_series)):
        if i < 2:
            mode_list.append("안전")
            continue
        two_weeks_ago = rsi_series.iloc[i - 2]
        one_week_ago = rsi_series.iloc[i - 1]

        if (
            (two_weeks_ago > 65 and two_weeks_ago > one_week_ago) or
            (40 < two_weeks_ago < 50 and two_weeks_ago > one_week_ago) or
            (one_week_ago < 50 and 50 < two_weeks_ago)
        ):
            mode = "안전"
        elif (
            (two_weeks_ago < 35 and two_weeks_ago < one_week_ago) or
            (50 < two_weeks_ago < 60 and two_weeks_ago < one_week_ago) or
            (one_week_ago > 50 and 50 > two_weeks_ago)
        ):
            mode = "공세"
        else:
            mode = mode_list[i - 1]
        mode_list.append(mode)
    return mode_list

def get_future_market_day(start_day, market_days, offset_days):
    market_days = market_days[market_days > start_day]
    if len(market_days) < offset_days:
        return None
    return market_days[offset_days - 1].date()

# ---------- 주문 추출 ----------
def extract_orders(df):
    sell_orders = []
    buy_orders = []

    for _, row in df.iterrows():
        if pd.notna(row['매도목표가']) and row['매도목표가'] > 0 and pd.isna(row['실제매도일']) and row['주문유형'] != "MOC":
            price = round(row['매도목표가'], 2)
            qty = int(row['매수량']) if pd.notna(row['매수량']) else 0
            if qty > 0:
                sell_orders.append(Order("매도", "LOC", price, qty))

        elif pd.isna(row['실제매도일']) and pd.notna(row['MOC매도일']) and row['주문유형'] == "MOC":
            price = round(row['매도목표가'], 2)
            qty = int(row['매수량']) if pd.notna(row['매수량']) else 0
            if qty > 0:
                sell_orders.append(Order("매도", "MOC", price, qty))

    if df.empty:
        return [], []
    
    last_row = df.iloc[-1]

    if pd.notna(last_row['LOC매수목표']) and pd.notna(last_row['목표량']):
        price = round(last_row['LOC매수목표'], 2)
        qty = int(last_row['목표량'])
        if qty > 0:
            buy_orders.append(Order("매수", "LOC", price, qty))

    return sell_orders, buy_orders

def calc_balance(row, prev_balance, sell_list):
    if not row.get("종가"):
        return None

    planned_buy = row.get("매수예정", 0) or 0
    trade_day = row.get("일자")

    today_sell_profit = sum(
        s.get("실제매도금액", 0)
        for s in sell_list
        if s.get("실제매도일") == trade_day
    )

    return round(prev_balance - planned_buy + today_sell_profit, 2)

# ---------------------------------------
# ✅ RSI 매매 전략 실행
# ---------------------------------------
def get_mode_and_target_prices(start_date, end_date, target_ticker, first_amt, day_cnt, safe_hold_days, safe_buy_threshold, safe_sell_threshold, aggr_hold_days, aggr_buy_threshold, aggr_sell_threshold, aggr_div_cnt, prft_cmpnd_int_rt, loss_cmpnd_int_rt):

    v_first_amt = first_amt
    result = []

    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    qqq_start = start_dt - pd.Timedelta(weeks=20)

    nyse = mcal.get_calendar("NYSE")
    market_days = nyse.schedule(
        start_date=qqq_start.strftime("%Y-%m-%d"),
        end_date=(end_dt + pd.Timedelta(days=safe_hold_days + 60)).strftime("%Y-%m-%d")
    ).index.normalize()
    
    qqq = fdr.DataReader("QQQ", qqq_start.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    qqq.index = pd.to_datetime(qqq.index)
    if end_dt not in qqq.index:
        qqq.loc[end_dt] = None

    weekly = get_last_trading_day_each_week(qqq)
    weekly_rsi = calculate_rsi_rolling(weekly).dropna(subset=["RSI"])
    weekly_rsi["모드"] = assign_mode_v2(weekly_rsi["RSI"])
    weekly_rsi["year"] = weekly_rsi.index.year
    weekly_rsi["week"] = weekly_rsi.index.map(get_weeknum_google_style)
    mode_by_year_week = weekly_rsi.set_index(["year", "week"])[["모드", "RSI"]]

    ticker_data = fdr.DataReader(target_ticker, qqq_start.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    ticker_data.index = pd.to_datetime(ticker_data.index)

    for day in market_days:
        if not (start_dt <= day <= end_dt):
            continue

        year, week = day.year, get_weeknum_google_style(day)
        if (year, week) not in mode_by_year_week.index:
            continue

        mode_info = mode_by_year_week.loc[(year, week)]
        mode = mode_info["모드"]
        rsi = round(mode_info["RSI"], 2)

        prev_days = ticker_data.index[ticker_data.index < day]
        if len(prev_days) == 0:
            continue
        prev_close = round(ticker_data.loc[prev_days[-1], "Close"], 2)

        actual_close = ticker_data.loc[day, "Close"] if day in ticker_data.index else None
        if pd.notna(actual_close):
            actual_close = round(actual_close, 2)
        today_close = actual_close

        if mode == "안전":
            div_cnt = safe_div_cnt
            target_price = round(prev_close * (1 + safe_buy_threshold), 2)
            sell_target_price = round((actual_close or target_price) * (1 + safe_sell_threshold), 2)
            holding_days = safe_hold_days
        else:
            div_cnt = aggr_div_cnt
            target_price = round(prev_close * (1 + aggr_buy_threshold), 2)
            sell_target_price = round((actual_close or target_price) * (1 + aggr_sell_threshold), 2)
            holding_days = aggr_hold_days

        daily_buy_amount = round(v_first_amt / div_cnt, 2)
        target_qty = int(daily_buy_amount // target_price) if target_price else 0

        buy_qty = 0
        buy_amt = None
        moc_sell_date = get_future_market_day(day, market_days, holding_days)
        
        actual_sell_date = actual_sell_price = actual_sell_qty = actual_sell_amount = prft_amt = None
        order_type = ""

        if actual_close and target_price >= actual_close and target_qty > 0:
            buy_qty = target_qty
            buy_amt = round(buy_qty * actual_close, 2)

            hold_range = market_days[(market_days >= day)][:holding_days]
            future_prices = ticker_data.loc[ticker_data.index.isin(hold_range)]
            match = future_prices[future_prices["Close"] >= sell_target_price]

            if not match.empty:
                actual_sell_date = match.index[0].date()
                actual_sell_price = round(match.iloc[0]["Close"], 2)
            elif moc_sell_date and pd.Timestamp(moc_sell_date) in ticker_data.index:
                actual_sell_date = moc_sell_date
                actual_sell_price = round(ticker_data.loc[pd.Timestamp(moc_sell_date)]["Close"], 2)

            if actual_sell_date:
                if actual_sell_date == moc_sell_date:
                    order_type = "MOC"
                else:
                    order_type = "LOC"
            else:
                order_type = "LOC"

        else:
            actual_close = None
            sell_target_price = None
            moc_sell_date = None
            prft_amt = 0.0

        result.append({
            "일자": day.date(),
            "종가": today_close,
            "모드": mode,
            "변동률": round((today_close - prev_close) / prev_close * 100, 2) if today_close and prev_close else np.nan,
            "매수예정": None,
            "LOC매수목표": target_price,
            "목표량": None,
            "매수가": actual_close,
            "매수량": None,
            "매수금액": None,
            "매도목표가": sell_target_price,
            "MOC매도일": moc_sell_date,
            "실제매도일": actual_sell_date,
            "실제매도가": actual_sell_price,
            "실제매도량": None,
            "실제매도금액": None,
            "당일실현": None,
            "매매손익": None,
            "누적매매손익": None,
            "복리금액": None,
            "자금갱신": None,
            "예수금": None,
            "주문유형": order_type
        })

        day_cnt += 1

    prev_cash = prev_pmt_update = first_amt
    prev_profit_sum = 0.0
    daily_realized_profits = {}

    for i, row in enumerate(result):
        if row["모드"] == "안전":
            div_cnt = safe_div_cnt
        else:
            div_cnt = aggr_div_cnt

        base_amt = round((prev_pmt_update if i > 0 else first_amt) / div_cnt, 2)
        if prev_cash is None:
            row["매수예정"] = base_amt
        else:
            row["매수예정"] = min(base_amt, prev_cash)

        tgt_price, buy_price, sell_price = row["LOC매수목표"], row["매수가"], row["실제매도가"]
        qty = int(row["매수예정"] // tgt_price) if tgt_price else None
        row["목표량"] = qty
        row["매수량"] = qty if buy_price else None
        row["매수금액"] = round(qty * buy_price, 2) if qty and buy_price else None

        if qty and sell_price:
            row["실제매도량"] = qty
            row["실제매도금액"] = round(qty * sell_price, 2)
            row["매매손익"] = row["실제매도금액"] - (row["매수금액"] or 0)

        if row["매매손익"] is not None:
            prev_profit_sum += row["매매손익"]

        row["누적매매손익"] = prev_profit_sum

        buy_amt = row.get("매수금액") or 0
        trade_day = row["일자"]
        sell_amt = sum(r.get("실제매도금액") or 0 for r in result if r["실제매도일"] == trade_day)
        prev_cash = prev_cash - buy_amt + sell_amt
        row["예수금"] = prev_cash if row["종가"] else None

        if trade_day not in daily_realized_profits:
            daily_realized_profits[trade_day] = sum((r.get("매매손익") or 0) for r in result if r.get("실제매도일") == trade_day)
        row["당일실현"] = daily_realized_profits[trade_day] or None

        if (i + 1) % INVT_RENWL_CYLE == 0:
            bfs = sum((r.get("당일실현") or 0) for r in result[max(0, i - INVT_RENWL_CYLE + 1):i + 1])
            rate = prft_cmpnd_int_rt if bfs > 0 else loss_cmpnd_int_rt
            row["복리금액"] = round(bfs * rate, 2)
        else:
            row["복리금액"] = None

        prev_pmt_update += row["복리금액"] or 0
        row["자금갱신"] = prev_pmt_update

    return pd.DataFrame(result)
    

# ----------상계 처리 표 출력 ----------
def print_table(orders):
    df = pd.DataFrame([{
        "매매유형": order.side,
        "주문유형": order.type,
        "주문가": round(order.price, 2),
        "수량": order.quantity
    } for order in orders])

    return df

def print_orders(sell_orders, buy_orders):
    print("\n---[매도 주문]")
    print(f"{'Side':<10}{'Type':<10}{'Price':<10}{'Quantity':<10}")
    print("-" * 40)
    for order in sorted(sell_orders, key=lambda x: x.price, reverse=True):
        print(f"{order.side:<10}{order.type:<10}{order.price:<10.2f}{order.quantity:<10}")

    print("\n---[매수 주문]")
    print(f"{'Side':<10}{'Type':<10}{'Price':<10}{'Quantity':<10}")
    print("-" * 40)
    for order in sorted(buy_orders, key=lambda x: x.price):
        print(f"{order.side:<10}{order.type:<10}{order.price:<10.2f}{order.quantity:<10}")

def remove_duplicates(sell_orders, buy_orders):
    if not sell_orders or not buy_orders:
        return

    buy_order = buy_orders[0]

    filtered_sell_orders = []
    new_sell_orders = []
    new_buy_orders = []

    sell_moc_order = None
    b_exist_moc = False

    for sell_order in sell_orders:
        if sell_order.type == "MOC":
            sell_moc_order = sell_order
            b_exist_moc = True
            continue

        if sell_order.price <= buy_order.price:
            filtered_sell_orders.append(sell_order)
        else:
            new_sell_orders.append(sell_order)

    if not b_exist_moc and not filtered_sell_orders:
        return

    buy_order_quantity = buy_order.quantity

    if b_exist_moc:
        if sell_moc_order.quantity > buy_order.quantity:
            new_sell_orders.append(Order("매도","MOC", 0.0, sell_moc_order.quantity - buy_order.quantity ))
            buy_order = buy_order._replace(quantity=0)
        elif sell_moc_order.quantity == buy_order.quantity:
            buy_order = buy_order._replace(quantity=0)
        else:
            buy_order = buy_order._replace(quantity=buy_order.quantity - sell_moc_order.quantity)
            if not filtered_sell_orders:
                new_sell_orders.append(Order("매도","LOC", round(buy_order.price + 0.01, 2), sell_moc_order.quantity))

    filtered_sell_orders.sort(key=lambda x: x.price)

    for sell_order in filtered_sell_orders:
        if buy_order.quantity == 0:
            new_sell_orders.append(sell_order)
            continue

        if sell_order.quantity >= buy_order.quantity:
            new_buy_orders.append(Order("매수","LOC", round(sell_order.price - 0.01, 2), buy_order.quantity))
            if sell_order.quantity > buy_order.quantity:
                new_sell_orders.append(Order("매도","LOC", round(sell_order.price, 2), sell_order.quantity - buy_order.quantity))
            buy_order = buy_order._replace(quantity=0)
        else:
            new_buy_orders.append(Order("매수","LOC", round(sell_order.price - 0.01, 2), sell_order.quantity))
            buy_order = buy_order._replace(quantity=buy_order.quantity - sell_order.quantity)

    if buy_order.quantity != 0:
        new_buy_orders.append(Order("매수","LOC", round(buy_order.price, 2), buy_order.quantity))
        sell_quant = sum(order.quantity for order in filtered_sell_orders)
        if sell_quant != 0:
            new_sell_orders.append(Order("매도","LOC", round(buy_order.price + 0.01, 2), sell_quant))
    else:
        new_sell_orders.append(Order("매도","LOC", round(buy_order.price + 0.01, 2), buy_order_quantity))

    new_sell_orders.sort(key=lambda x: x.price, reverse=True)
    new_buy_orders.sort(key=lambda x: x.price, reverse=True)

    sell_orders[:] = new_sell_orders
    buy_orders[:] = new_buy_orders

def highlight_order(row):
    if row["매매유형"] == "매도":
        return ['background-color: #D9EFFF'] * len(row)
    elif row["매매유형"] == "매수":
        return ['background-color: #FFE6E6'] * len(row)
    else:
        return [''] * len(row)
    
# ---------------------------------------
# ✅ Streamlit UI
# ---------------------------------------
st.title("📈 RSI 변동성 매매")

# ---------------------------------------
# ✅ 설정 로드 (사용자 이름)
# ---------------------------------------
config = load_config()
user_names = config["user_names"]

# ---------------------------------------
# ✅ 사이드바에 사용자 이름 관리 섹션 추가
# ---------------------------------------
st.sidebar.subheader("👨‍💻 사용자 이름 관리")
new_user_names = []
for i, name in enumerate(user_names):
    new_name = st.sidebar.text_input(f"사용자 {i+1} 이름", value=name)
    new_user_names.append(new_name)

if st.sidebar.button("사용자 이름 저장"):
    config["user_names"] = new_user_names
    save_config(config)
    st.sidebar.success("사용자 이름이 저장되었습니다!")
    st.rerun()

# ---------------------------------------
# ✅ 사용자 선택 드롭다운
# ---------------------------------------
st.subheader("👨‍💻 사용자 선택")
if 'selected_user_name' not in st.session_state or st.session_state.selected_user_name not in user_names:
    st.session_state.selected_user_name = user_names[0]

selected_user = st.selectbox("사용자 이름", user_names, index=user_names.index(st.session_state.selected_user_name))

if selected_user != st.session_state.selected_user_name:
    st.session_state.selected_user_name = selected_user
    st.rerun()

# 선택된 사용자의 파라미터 로드
params = load_params(st.session_state.selected_user_name)

# ---------------------------------------
# 스타일 설정 사전
# ---------------------------------------
styles = {
    "Default": {
        "safe_hold_days": 30,
        "safe_buy_threshold": 3.0,
        "safe_sell_threshold": 0.2,
        "safe_div_cnt": 7,
        "aggr_hold_days": 7,
        "aggr_buy_threshold": 5.0,
        "aggr_sell_threshold": 2.5,
        "aggr_div_cnt": 7,
        "prft_cmpnd_int_rt": 0.8,
        "loss_cmpnd_int_rt": 0.3,
    },
    "공격형2": {
        "safe_hold_days": 35,
        "safe_buy_threshold": 3.5,
        "safe_sell_threshold": 1.8,
        "safe_div_cnt": 7,
        "aggr_hold_days": 7,
        "aggr_buy_threshold": 3.6,
        "aggr_sell_threshold": 5.6,
        "aggr_div_cnt": 8,
        "prft_cmpnd_int_rt": 0.72,
        "loss_cmpnd_int_rt": 0.213,
    }
}

# ---------------------------------------
# 공통 파라미터
# ---------------------------------------
st.subheader("💹 공통 항목 설정")

# 📝 스타일 선택
style_option = st.selectbox("스타일 선택", list(styles.keys()), index=list(styles.keys()).index(params["style_option"]))
selected_style = styles[style_option]
if style_option != params["style_option"]:
    params["style_option"] = style_option
    save_params(params, st.session_state.selected_user_name)

col1, col2 = st.columns(2)

with col1:
    # 📝 티커 선택
    tickers = ('SOXL', 'KORU', 'TQQQ', 'BITU')
    target_ticker = st.selectbox('티커 *', tickers, index=tickers.index(params["target_ticker"]))
    if target_ticker != params["target_ticker"]:
        params["target_ticker"] = target_ticker
        save_params(params, st.session_state.selected_user_name)

with col2:
    # 📝 투자금액 입력
    first_amt = st.number_input("투자금액(USD) *", value=params["first_amt"], step=500)
    if first_amt != params["first_amt"]:
        params["first_amt"] = first_amt
        save_params(params, st.session_state.selected_user_name)
    st.markdown(f"**입력한 투자금액:** {first_amt:,}")

# 시작일자 + 종료일자
col3, col4 = st.columns(2)

with col3:
    # 📝 투자 시작일 입력
    start_date = st.date_input("투자시작일 *", value=datetime.strptime(params["start_date"], '%Y-%m-%d').date())
    if start_date.strftime('%Y-%m-%d') != params["start_date"]:
        params["start_date"] = start_date.strftime('%Y-%m-%d')
        save_params(params, st.session_state.selected_user_name)

with col4:
    # 📝 투자 종료일 입력
    end_date = st.date_input("투자종료일 *", value=datetime.strptime(params["end_date"], '%Y-%m-%d').date())
    # ⛔️ 수정된 부분: 아래 두 줄을 삭제 또는 주석 처리하여 투자 종료일이 저장되지 않도록 함
    # if end_date.strftime('%Y-%m-%d') != params["end_date"]:
    #     params["end_date"] = end_date.strftime('%Y-%m-%d')
    #     save_params(params, st.session_state.selected_user_name)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------
# 안전모드 파라미터
# ---------------------------------------
st.subheader("💹 안전모드 설정")
safe_hold_days = selected_style["safe_hold_days"]
safe_buy_threshold = selected_style["safe_buy_threshold"] / 100
safe_sell_threshold = selected_style["safe_sell_threshold"] / 100
safe_div_cnt = selected_style["safe_div_cnt"]

st.markdown(f"**최대보유일수:** {safe_hold_days}일")
st.markdown(f"**분할수:** {safe_div_cnt}회")

col5, col6 = st.columns(2)
with col5:
    st.markdown(f"**매수조건이율:** {selected_style['safe_buy_threshold']}%")

with col6:
    st.markdown(f"**매도조건이율:** {selected_style['safe_sell_threshold']}%")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------
# 공세모드 파라미터
# ---------------------------------------
st.subheader("💹 공세모드 설정")
aggr_hold_days = selected_style["aggr_hold_days"]
aggr_buy_threshold = selected_style["aggr_buy_threshold"] / 100
aggr_sell_threshold = selected_style["aggr_sell_threshold"] / 100
aggr_div_cnt = selected_style["aggr_div_cnt"]

st.markdown(f"**최대보유일수:** {aggr_hold_days}일")
st.markdown(f"**분할수:** {aggr_div_cnt}회")

col7, col8 = st.columns(2)
with col7:
    st.markdown(f"**매수조건이율:** {selected_style['aggr_buy_threshold']}%")

with col8:
    st.markdown(f"**매도조건이율:** {selected_style['aggr_sell_threshold']}%")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("▶ 전략 실행"):
    status_placeholder = st.empty()
    status_placeholder.info("전략 실행 중입니다...")

    prft_cmpnd_int_rt = selected_style["prft_cmpnd_int_rt"]
    loss_cmpnd_int_rt = selected_style["loss_cmpnd_int_rt"]

    df_result = get_mode_and_target_prices(start_date, end_date, target_ticker, first_amt, 0, safe_hold_days, safe_buy_threshold, safe_sell_threshold, aggr_hold_days, aggr_buy_threshold, aggr_sell_threshold, aggr_div_cnt, prft_cmpnd_int_rt, loss_cmpnd_int_rt)

    pd.set_option('future.no_silent_downcasting', True)
    printable_df = df_result.replace({None: np.nan})
    printable_df = printable_df.astype(str).replace({"None": "", "nan": ""})

    if printable_df.empty:
        status_placeholder.empty()
        st.warning("데이터가 없습니다. 입력 조건을 확인하세요.")
    else:
        status_placeholder.empty()
        st.success("전략 실행 완료!")

        buy_data = df_result[["일자", "매수가", "매수량"]].copy()
        buy_data.columns = ["date", "price", "quantity"]
        sell_data = df_result[["실제매도일", "실제매도가", "실제매도량"]].copy()
        sell_data.columns = ["date", "price", "quantity"]
        sell_data = sell_data.dropna(subset=["quantity"])
        sell_data["quantity"] = -sell_data["quantity"]
        df = pd.concat([buy_data, sell_data], ignore_index=True)
        df = df.dropna(subset=["date", "price", "quantity"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        avg_prc = 0
        history = []
        unique_dates = df["date"].unique()

        for date in unique_dates:
            sub = df[df["date"] == date]
            p = sub["price"].iloc[0]
            q = sub["quantity"].sum()
            past_qty = df[df["date"] < date]["quantity"].sum()

            if avg_prc == 0:
                avg_prc = p
            elif q < 0:
                pass
            else:
                avg_prc = (avg_prc * past_qty + p * q) / (past_qty + q)
            history.append((date.date(), round(avg_prc, 4)))

        total_qty = int(df["quantity"].sum())
        total_profit = df_result.dropna(subset=["실제매도금액", "매수금액"]).apply(
            lambda row: (row["실제매도금액"] - row["매수금액"]), axis=1
        ).sum()
        profit_ratio = (total_profit / first_amt * 100)

        st.markdown("<br>", unsafe_allow_html=True)
        
        summary_data = {
            "항목": [
                "📦 현재 보유량",
                "💵 평균 단가",
                "📈 누적 매매손익",
                "📊 수익률(누적매매손익 / 투자원금)"
            ],
            "값": [
                f"{total_qty:,} 주",
                f"${avg_prc:,.2f}",
                f"${total_profit:,.2f}",
                f"{profit_ratio:.2f} %"
            ]
        }
        summary_df = pd.DataFrame(summary_data)

        st.subheader("💹 요 약")
        st.table(summary_df)

        styled_df = printable_df.style.format({
            "종가": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "변동률": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "매수예정": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "LOC매수목표": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "목표량": lambda x: "{:.0f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "매수가": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "매수량": lambda x: "{:.0f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "매수금액": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "매도목표가": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "실제매도가": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "실제매도량": lambda x: "{:.0f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "실제매도금액": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "당일실현": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "매매손익": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "누적매매손익": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "복리금액": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "자금갱신": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "예수금": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
        })

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 매매 리스트")
        st.dataframe(styled_df)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_result.to_excel(writer, sheet_name="매매리스트", index=False)
        excel_data = output.getvalue()

        st.download_button(
            label="⬇️ 엑셀 다운로드",
            data=excel_data,
            file_name="strategy_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    sell_orders, buy_orders = extract_orders(df_result)
    print_orders(sell_orders, buy_orders)
    remove_duplicates(sell_orders, buy_orders)

    df_sell = print_table(sell_orders)
    df_buy = print_table(buy_orders)
    df_result = pd.concat([df_sell, df_buy], ignore_index=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 당일 주문 리스트")
    styled_df = (df_result
                     .style
                     .apply(highlight_order, axis=1).format({"주문가": "{:.2f}"})
                ) 
    st.dataframe(styled_df, use_container_width=True)
    
