import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from collections import namedtuple
import numpy as np
import FinanceDataReader as fdr

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
    return int(date.strftime("%Y%U"))  # %U: 주차 (일요일 시작)

# ---------------------------------------
# ✅ 주요 파라미터 (전략 설정값)
# ---------------------------------------
DIV_CNT = 7                        # 분할횟수

# 안전모드 설정
SAFE_BUY_THRESHOLD = 0.03          # 안전모드 매수조건이율
SAFE_SELL_THRESHOLD = 0.002        # 안전모드 매도조건이율
SAFE_HOLD_DAYS = 30                # 안전모드 최대보유일수

# 공세모드 설정
AGGR_BUY_THRESHOLD = 0.05          # 공세모드 매수조건이율
AGGR_SELL_THRESHOLD = 0.025        # 공세모드 매도조건이율
AGGR_HOLD_DAYS = 7                 # 공세모드 최대보유일수


# 투자금 갱신 설정
PRFT_CMPND_INT_RT = 0.8            # 이익복리율
LOSS_CMPND_INT_RT = 0.3            # 손실복리율
INVT_RENWL_CYLE   = 10             # 투자금갱신주기

# 주문 정보 구조 정의
Order = namedtuple('Order', ['side', 'type', 'price', 'quantity'])

# ---------- 유틸 함수들 ----------
def get_weeknum_google_style(date):
    """
    Google Calendar 스타일의 주차(Week Number) 계산
    기준: 1월 1일부터 시작, 요일 보정 포함
    """    
    jan1 = pd.Timestamp(year=date.year, month=1, day=1).tz_localize(None)
    date = pd.Timestamp(date).tz_localize(None)
    weekday_jan1 = jan1.weekday()
    delta_days = (date - jan1).days
    return ((delta_days + weekday_jan1) // 7) + 1

def get_last_trading_day_each_week(data):
    """
    각 주차별로 가장 마지막 거래일 데이터를 추출 (주간 RSI 계산용)
    """    
    data = data.copy()
    #print("---data :", data)    
    data['week'] = data.index.to_series().apply(get_weeknum_google_style)
    data['year'] = data.index.to_series().dt.year
    data['weekday'] = data.index.to_series().dt.weekday
    last_day = data.groupby(['year', 'week'])[['weekday']].idxmax()
    #print("---last_day :", last_day)
    #print("---end last_day ")    
    return data.loc[last_day['weekday']]

def calculate_rsi_rolling(data, period=14):
    """
    RSI(상대강도지수)를 주어진 기간 기준으로 계산
    기본: 14일
    """    
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
    """
    RSI 흐름을 기반으로 안전/공세 모드를 판별
    2주 전과 1주 전 RSI 값을 비교
    """    
    mode_list = []
    for i in range(len(rsi_series)):
        if i < 2:
            mode_list.append("안전") # 초기에는 무조건 안전모드
            continue
        two_weeks_ago = rsi_series.iloc[i - 2]
        one_week_ago = rsi_series.iloc[i - 1]

        # 안전 조건        
        if (
            (two_weeks_ago > 65 and two_weeks_ago > one_week_ago) or
            (40 < two_weeks_ago < 50 and two_weeks_ago > one_week_ago) or
            (one_week_ago < 50 and 50 < two_weeks_ago)
        ):
            mode = "안전"
        # 공세 조건            
        elif (
            (two_weeks_ago < 35 and two_weeks_ago < one_week_ago) or
            (50 < two_weeks_ago < 60 and two_weeks_ago < one_week_ago) or
            (one_week_ago > 50 and 50 > two_weeks_ago)
        ):
            mode = "공세"
        else:
            mode = mode_list[i - 1]  # 이전 모드를 유지
        mode_list.append(mode)
    return mode_list

def get_future_market_day(start_day, market_days, offset_days):
    """
    기준일로부터 N일 후의 거래일 반환
    예: MOC 매도를 위한 미래 보유일 확인
    """    
    market_days = market_days[market_days > start_day]
    if len(market_days) < offset_days:
        return None
    return market_days[offset_days - 1].date()

# ---------- 주문 추출 ----------
def extract_orders(df):
    """
    DataFrame에서 매수/매도 대상 주문 추출
    - 매도: 목표가 존재하고 아직 매도되지 않은 건
    - 매수: 마지막 날 LOC 매수 목표가 존재하는 경우
    """    
    sell_orders = []
    buy_orders = []

    for _, row in df.iterrows():
        if pd.notna(row['매도목표가']) and row['매도목표가'] > 0 and pd.isna(row['실제매도일']) and row['주문유형'] != "MOC":              
            price = round(row['매도목표가'], 2)
            qty = int(row['매수량']) if pd.notna(row['매수량']) else 0
            if qty > 0:
                sell_orders.append(Order("Sell", "LOC", price, qty))
               # print("----->>>>> sell_orders1 : ", sell_orders)

        # 실제매도일이 미입력이고 MOC매도일이 존재하고 주문유형이 MOC일 경우        
        elif pd.isna(row['실제매도일']) and pd.notna(row['MOC매도일']) and row['주문유형'] == "MOC":                        
            price = round(row['매도목표가'], 2)
            qty = int(row['매수량']) if pd.notna(row['매수량']) else 0
            if qty > 0:
                sell_orders.append(Order("Sell", "MOC", price, qty))
                #print("----->>>>> sell_orders2 : ", sell_orders)                

    if df.empty:
        return [], []
    
    last_row = df.iloc[-1]

    if pd.notna(last_row['LOC매수목표']) and pd.notna(last_row['목표량']):
        price = round(last_row['LOC매수목표'], 2)
        qty = int(last_row['목표량'])
        if qty > 0:
            buy_orders.append(Order("Buy", "LOC", price, qty))
            #print("----->>>>> buy_orders1 : ", buy_orders)            

    #print("----->>>>> sell_orders9 : ", sell_orders)
    return sell_orders, buy_orders

# ---------------------------------------
# ✅ RSI 매매 전략 실행
# ---------------------------------------
# ---------- 매매 전략 실행 ----------
def get_mode_and_target_prices(start_date, end_date, target_ticker, first_amt):
    """
    입력된 기간 동안의 전략 모드(안전/공세)와 매매 목표가/실제 체결 내역을 계산

    Parameters:
        start_date (str): 시작일 (예: '2025-06-01')
        end_date (str): 종료일 (예: '2025-06-20')

    Returns:
        pd.DataFrame: 날짜별 매매 전략, 매수/매도 목표가, 체결가, 체결 수량 등 포함된 결과표
    """    

    daily_buy_amount = round(first_amt / DIV_CNT, 2)  # 1회 매수에 사용할 금액

    # 날짜 전처리
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    qqq_start = start_dt - pd.Timedelta(weeks=20)  # RSI 계산을 위한 20주치 데이터 필요

    # 거래일 계산 (미국장 기준)
    nyse = mcal.get_calendar("XNYS")
    all_days = nyse.schedule(
        start_date=qqq_start.strftime("%Y-%m-%d"),
        end_date=(end_dt + pd.Timedelta(days=SAFE_HOLD_DAYS + 10)).strftime("%Y-%m-%d")
    )
    market_days = all_days.index.normalize()

    # QQQ 데이터 FDR로 가져오기
    qqq = fdr.DataReader("QQQ", qqq_start.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    qqq.index = pd.to_datetime(qqq.index)

    weekly = get_last_trading_day_each_week(qqq)
    weekly_rsi = calculate_rsi_rolling(weekly).dropna(subset=["RSI"])
    weekly_rsi['모드'] = assign_mode_v2(weekly_rsi['RSI'])
    weekly_rsi['year'] = weekly_rsi.index.to_series().dt.year
    weekly_rsi['week'] = weekly_rsi.index.to_series().apply(get_weeknum_google_style)
    weekly_rsi['rsi_date'] = weekly_rsi.index.date
    mode_by_year_week = weekly_rsi.set_index(['year', 'week'])[['모드', 'RSI', 'rsi_date']]
    #print("mode_by_year_week :", mode_by_year_week)

    # SOXL 데이터 FDR로 가져오기
    soxl = fdr.DataReader(target_ticker, qqq_start.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    soxl.index = pd.to_datetime(soxl.index)

    result = []

    # 각 거래일마다 전략 수립    
    for day in market_days:
        if day < start_dt or day > end_dt:
            continue

        # 해당 날짜의 연도 및 주차 정보로 모드(RSI 기반) 조회
        year = day.year
        week = get_weeknum_google_style(day)

        if (year, week) not in mode_by_year_week.index:
            continue

        row = mode_by_year_week.loc[(year, week)]
        mode = row['모드']
        rsi = round(row['RSI'], 2)
        rsi_date = row['rsi_date']

        # 전일 종가 조회
        prev_days = soxl.index[soxl.index < day]
        if len(prev_days) == 0:
            continue
        
        prev_close = round(soxl.loc[prev_days[-1], 'Close'], 2)

        # 해당일 종가 (체결 여부 판단용)
        actual_close = soxl.loc[day, 'Close'] if day in soxl.index else None
        if pd.isna(actual_close):
            actual_close = None
        if actual_close is not None:
            actual_close = round(actual_close, 2)

        today_close = actual_close #  당일 종가 화면 출력용

        # 모드에 따라 목표가 및 보유일 설정
        if mode == "안전":
            target_price = round(prev_close * (1 + SAFE_BUY_THRESHOLD), 2)
            sell_target_price = round((actual_close or target_price) * (1 + SAFE_SELL_THRESHOLD), 2)
            holding_days = SAFE_HOLD_DAYS
        else:
            target_price = round(prev_close * (1 + AGGR_BUY_THRESHOLD), 2)
            sell_target_price = round((actual_close or target_price) * (1 + AGGR_SELL_THRESHOLD), 2)
            holding_days = AGGR_HOLD_DAYS

        # 목표 수량 계산
        target_qty = int(daily_buy_amount // target_price)
        actual_qty = int(daily_buy_amount // target_price) if actual_close else None
        buy_amt = round(actual_qty * actual_close, 2) if actual_qty and actual_close else None

        # MOC 매도일 = 보유일 후 첫 거래일
        moc_sell_date = get_future_market_day(day, market_days, holding_days)

        # 초기화: 실제 매도 관련 정보
        actual_sell_date = None
        actual_sell_price = None
        actual_sell_qty = None
        actual_sell_amount = None
        order_type = ""

        # 실제 체결 가능한 경우 (매수 목표가 ≥ 종가)
        if actual_close and target_price >= actual_close:
            # 보유 기간 내 종가가 매도 목표가를 넘긴 경우 매도 성사            
            hold_range = market_days[(market_days >= day)][:holding_days]
            future_prices = soxl.loc[soxl.index.isin(hold_range)]

            match = future_prices[future_prices['Close'] >= sell_target_price]
            if not match.empty:
                actual_sell_date = match.index[0].date()
                actual_sell_price = round(match.iloc[0]['Close'], 2)
            elif moc_sell_date and pd.Timestamp(moc_sell_date) in soxl.index:
                # 조건 달성 실패 시 MOC 매도                
                actual_sell_date = moc_sell_date
                actual_sell_price = round(soxl.loc[pd.Timestamp(moc_sell_date)]['Close'], 2)

            if actual_sell_price:
                actual_sell_qty = actual_qty
                actual_sell_amount = round(actual_sell_price * actual_sell_qty, 2)

            if not actual_sell_date:
                # 당일이 MOC 매도일이라면 MOC로 판별                         
                if moc_sell_date == end_dt.date():
                    order_type = "MOC"
                    #print("---- MOC-----")
                else:
                    order_type = "LOC"
            else:
                #  MOC 매도일과 살제매도일이 같으면 MOC
                if moc_sell_date == actual_sell_date:
                    order_type = "MOC"
                else:               	                    	
                    order_type = "LOC"  

        # 매수 미체결 시: 관련 값 모두 초기화
        elif actual_close is not None and target_price < actual_close:
            #print("모드 존재 여부2:", week, day)            
            actual_close = None
            actual_qty = None
            buy_amt = None
            sell_target_price = None
            moc_sell_date = None
            actual_sell_date = None
            actual_sell_price = None
            actual_sell_qty = None
            actual_sell_amount = None
            order_type = ""

        # 결과 누적
        result.append({
            "일자": day.date(),
            "모드": mode,
            #"RSI일자": rsi_date,
            #"RSI": rsi,
            "종가": today_close,
            "변동률": round((today_close - prev_close) / prev_close * 100, 2)
            if isinstance(today_close, (int, float)) and prev_close else np.nan,            
            "매수예정": daily_buy_amount,
            "LOC매수목표": target_price,
            "목표량": target_qty,
            "매수가": actual_close,
            "매수량": actual_qty, 
            "매수금액": buy_amt,
            "매도목표가": sell_target_price,
            "MOC매도일": moc_sell_date,
            "실제매도일": actual_sell_date,
            "실제매도가": actual_sell_price,
            "실제매도량": actual_sell_qty,
            "실제매도금액": actual_sell_amount,
            "매매손익": round(actual_sell_amount - buy_amt, 2) if actual_sell_amount else None,
            "주문유형": order_type
        })

    return pd.DataFrame(result)

# ----------퉁치기 표 출력 ----------
def print_table(orders):
    """
    주문 리스트를 DataFrame으로 변환
    """
    df = pd.DataFrame([{
        "매매유형": order.side,
        "주문유형": order.type,
        "주문가": round(order.price, 2),
        "수량": order.quantity
    } for order in orders])

    #print("--- df : ", df)
    return df

#-- 매도/매수 주문내역 출력
def print_orders(sell_orders, buy_orders):
    """
    매도/매수 주문을 구분 출력
    - 매도는 가격 내림차순
    - 매수는 가격 오름차순
    """    
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


# ---------- 퉁치기 로직 ----------
def remove_duplicates(sell_orders, buy_orders):
    """
    LOC/MOC 주문을 기준으로 매수/매도 간 가격 정산 및 충돌 제거
    - 매도 주문은 가격 내림차순, 매수 주문은 오름차순 정렬
    - LOC 매수 가격보다 낮은 매도 주문은 퉁치기 후보
    """    
    if not sell_orders or not buy_orders:
        return

    buy_order = buy_orders[0]

    filtered_sell_orders = []
    new_sell_orders = []
    new_buy_orders = []

    sell_moc_order = None
    b_exist_moc = False

    # MOC 매도 주문과 LOC 매도 주문 분리
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

    # MOC 매도 주문 처리
    if b_exist_moc:
        if sell_moc_order.quantity > buy_order.quantity:
            new_sell_orders.append(Order("Sell","MOC", 0.0, sell_moc_order.quantity - buy_order.quantity ))
            buy_order = buy_order._replace(quantity=0)
        elif sell_moc_order.quantity == buy_order.quantity:
            buy_order = buy_order._replace(quantity=0)
        else:
            buy_order = buy_order._replace(quantity=buy_order.quantity - sell_moc_order.quantity)
            if not filtered_sell_orders:
                new_sell_orders.append(Order("Sell","LOC", round(buy_order.price + 0.01, 2), sell_moc_order.quantity))

    filtered_sell_orders.sort(key=lambda x: x.price)

    # LOC 매도 주문 퉁치기
    for sell_order in filtered_sell_orders:
        if buy_order.quantity == 0:
            new_sell_orders.append(sell_order)
            continue

        if sell_order.quantity >= buy_order.quantity:
            new_buy_orders.append(Order("Buy","LOC", round(sell_order.price - 0.01, 2), buy_order.quantity))
            if sell_order.quantity > buy_order.quantity:
                new_sell_orders.append(Order("Sell","LOC", round(sell_order.price, 2), sell_order.quantity - buy_order.quantity))
            buy_order = buy_order._replace(quantity=0)
        else:
            new_buy_orders.append(Order("Buy","LOC", round(sell_order.price - 0.01, 2), sell_order.quantity))
            buy_order = buy_order._replace(quantity=buy_order.quantity - sell_order.quantity)

    if buy_order.quantity != 0:
        new_buy_orders.append(Order("Buy","LOC", round(buy_order.price, 2), buy_order.quantity))
        sell_quant = sum(order.quantity for order in filtered_sell_orders)
        if sell_quant != 0:
            new_sell_orders.append(Order("Sell","LOC", round(buy_order.price + 0.01, 2), sell_quant))
    else:
        new_sell_orders.append(Order("Sell","LOC", round(buy_order.price + 0.01, 2), buy_order_quantity))

    new_sell_orders.sort(key=lambda x: x.price, reverse=True)
    new_buy_orders.sort(key=lambda x: x.price, reverse=True)

    sell_orders[:] = new_sell_orders
    buy_orders[:] = new_buy_orders

# ----- 퉁치기 표 색상 지정
def highlight_order(row):
    if row["매매유형"] == "Sell":
        return ['background-color: #D9EFFF'] * len(row)  # 하늘색
    elif row["매매유형"] == "Buy":
        return ['background-color: #FFE6E6'] * len(row)  # 분홍색
    else:
        return [''] * len(row)
    
# ---------------------------------------
# ✅ Streamlit UI
# ---------------------------------------
st.title("📈 RSI 동적 매매")

# 티커명
##ticker_input = st.text_input("투자 티커", value="SOXL")
##target_ticker = ticker_input.upper()

# target_ticker = st.selectbox('티커 ', ('SOXL','KORU','TQQQ','BITU'))

# # 최초투자금액
# first_amt = st.number_input("투자금액", value=20000, step=500)

# # 표시용 콤마 포맷 (예: 20,0000)
# st.markdown(f"**입력한 투자금액:** {first_amt:,}")

# start_date = st.date_input("시작일자", value= datetime.today() - timedelta(days=14))
# end_date = st.date_input("종료일자", value=datetime.today())

from datetime import datetime, timedelta
import streamlit as st

# 첫 번째 줄: 티커 선택 + 투자금액 입력
col1, col2 = st.columns(2)

with col1:
    target_ticker = st.selectbox('티커', ('SOXL', 'KORU', 'TQQQ', 'BITU'))

with col2:
    first_amt = st.number_input("투자금액", value=20000, step=500)
    st.markdown(f"**입력한 투자금액:** {first_amt:,}")

# 두 번째 줄: 시작일자 + 종료일자
col3, col4 = st.columns(2)

with col3:
    start_date = st.date_input("시작일자", value=datetime.today() - timedelta(days=14))

with col4:
    end_date = st.date_input("종료일자", value=datetime.today())


if st.button("▶ 전략 실행"):
    status_placeholder = st.empty()
    status_placeholder.info("전략 실행 중입니다...")

    df_result = get_mode_and_target_prices(start_date, end_date, target_ticker, first_amt)

    # NaN 및 None 값을 빈 문자열로 대체하여 출력
    #printable_df = df_result.where(pd.notnull(df_result), "")
    printable_df = df_result.replace({None: np.nan})
    printable_df = printable_df.astype(str).replace({"None": "", "nan": ""})

    if printable_df.empty:
        status_placeholder.empty()
        st.warning("데이터가 없습니다. 입력 조건을 확인하세요.")
    else:
        status_placeholder.empty()
        st.success("전략 실행 완료!")

        # ✅ 누적 매매손익
        total_profit = df_result.dropna(subset=["실제매도금액", "매수금액"]).apply(
            lambda row: (row["실제매도금액"] - row["매수금액"]), axis=1
        ).sum()

        #total_invested = df_result.dropna(subset=["매수금액"]).apply(
        #    lambda row: row["매수금액"], axis=1
        #).sum()
        
        # 수익률(누적매매손익 / 투자원금)
        #profit_ratio = (total_profit / total_invested * 100) if total_invested else 0
        profit_ratio = (total_profit / first_amt * 100)

        # 빈 줄 추가
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 💹 누적매매손익 & 수익률 표시
        col1, col2 = st.columns(2)
        col1.metric("📈 누적 매매손익", f"{total_profit:,.2f} USD")
        col2.metric("📊 수익률(누적매매손익 / 투자원금)", f"{profit_ratio:.2f} %")

        #print("----1111> :", printable_df.isnull().sum())

        # lambda에서 Null 아 아니고 숫자 아닌 경우 빈값으로 처리
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
            "매매손익": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
        })

        st.subheader("📊 매매 리스트")

        st.dataframe(styled_df)

        csv = df_result.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

        st.download_button("⬇️ CSV 다운로드", csv, "strategy_result.csv", "text/csv")

    # 퉁치기 대상 주문 추출
    sell_orders, buy_orders = extract_orders(df_result)
    print_orders(sell_orders, buy_orders)
    
    # 퉁치기
    remove_duplicates(sell_orders, buy_orders)

    df_sell = print_table(sell_orders)
    df_buy = print_table(buy_orders)
    df_result = pd.concat([df_sell, df_buy], ignore_index=True)
  
    # 빈 줄 추가
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("📊 당일 주문 리스트")
    styled_df = (df_result
                 .style
                 .apply(highlight_order, axis=1).format({"주문가": "{:.2f}"})
                ) 
    st.dataframe(styled_df, use_container_width=True)
    
