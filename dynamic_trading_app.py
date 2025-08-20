import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from collections import namedtuple
import numpy as np
import FinanceDataReader as fdr
import io

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
# DIV_CNT = 7                        # 분할횟수

# # 안전모드 설정
# SAFE_BUY_THRESHOLD = 0.03          # 안전모드 매수조건이율
# SAFE_SELL_THRESHOLD = 0.002        # 안전모드 매도조건이율
# SAFE_HOLD_DAYS = 30                # 안전모드 최대보유일수

# # 공세모드 설정
# AGGR_BUY_THRESHOLD = 0.05          # 공세모드 매수조건이율
# AGGR_SELL_THRESHOLD = 0.025        # 공세모드 매도조건이율
# AGGR_HOLD_DAYS = 7                 # 공세모드 최대보유일수


# 투자금 갱신 설정
# 복리 투자를 위해 필요하나 이 프로그램에서는 아직 반영하지 않았음
# 투자금갱신주기마다 일투자금액에 손실을 반영해서 투자하기 위함
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
    #print("---data['weekday']  :", data['weekday'] )
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
    예: MOC 매도를 위한 MOC매도일자 계산
    """    
    market_days = market_days[market_days > start_day]

    ##print("--> market_days1 : ", market_days)
    if len(market_days) < offset_days:
        return None
    
    ##print("--> market_days2 : ", start_day, market_days[offset_days - 1].date())
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
                sell_orders.append(Order("매도", "LOC", price, qty))
               # print("----->>>>> sell_orders1 : ", sell_orders)

        # 실제매도일이 미입력이고 MOC매도일이 존재하고 주문유형이 MOC일 경우        
        elif pd.isna(row['실제매도일']) and pd.notna(row['MOC매도일']) and row['주문유형'] == "MOC":                        
            price = round(row['매도목표가'], 2)
            qty = int(row['매수량']) if pd.notna(row['매수량']) else 0
            if qty > 0:
                sell_orders.append(Order("매도", "MOC", price, qty))
                #print("----->>>>> sell_orders2 : ", sell_orders)                

    if df.empty:
        return [], []
    
    last_row = df.iloc[-1]

    if pd.notna(last_row['LOC매수목표']) and pd.notna(last_row['목표량']):
        price = round(last_row['LOC매수목표'], 2)
        qty = int(last_row['목표량'])
        if qty > 0:
            buy_orders.append(Order("매수", "LOC", price, qty))
            #print("----->>>>> buy_orders1 : ", buy_orders)            

    return sell_orders, buy_orders

def calc_balance(row, prev_balance, sell_list):
    if not row.get("종가"):
        return None

    # 매수금액이 아닌 매수예정 기준으로 차감
    planned_buy = row.get("매수예정", 0) or 0
    trade_day = row.get("일자")

    today_sell_profit = sum(
        s.get("실제매도금액", 0)
        for s in sell_list
        if s.get("실제매도일") == trade_day
    )

    return round(prev_balance - planned_buy + today_sell_profit, 2)


# def calc_day_sell_profit(trade_day, sell_list):
#     return sum(
#         s.get("실제매도금액", 0) or 0
#         for s in sell_list
#         if s.get("실제매도일") == trade_day
#     )


# ---------------------------------------
# ✅ RSI 매매 전략 실행
# ---------------------------------------
# ---------- 매매 전략 실행 ----------
def get_mode_and_target_prices(start_date, end_date, target_ticker, first_amt, day_cnt):
    acum_prft_amt = 0.0
    v_first_amt = first_amt
    result = []

    # --- 날짜 및 RSI 모드 계산 ---
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    qqq_start = start_dt - pd.Timedelta(weeks=20)

    nyse = mcal.get_calendar("NYSE")
    market_days = nyse.schedule(
        start_date=qqq_start.strftime("%Y-%m-%d"),
        end_date=(end_dt + pd.Timedelta(days=safe_hold_days + 10)).strftime("%Y-%m-%d")
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

    # --- 매매 전략 수행 ---
    for day in market_days:
        if not (start_dt <= day <= end_dt):
            continue

        # # 10일마다 자본 업데이트
        # if day_cnt >= 10 and day_cnt % 10 == 0:
        #     v_first_amt += acum_prft_amt * 0.8

        daily_buy_amount = round(v_first_amt / div_cnt, 2)

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
            target_price = round(prev_close * (1 + safe_buy_threshold), 2)
            sell_target_price = round((actual_close or target_price) * (1 + safe_sell_threshold), 2)
            holding_days = safe_hold_days
        else:
            target_price = round(prev_close * (1 + aggr_buy_threshold), 2)
            sell_target_price = round((actual_close or target_price) * (1 + aggr_sell_threshold), 2)
            holding_days = aggr_hold_days

        # --- 매수 수량/금액: 한 번만 계산해서 '진실의 원천'으로 사용 ---
        target_qty = int(daily_buy_amount // target_price) if target_price else 0

        buy_qty = 0
        buy_amt = None
        moc_sell_date = get_future_market_day(day, market_days, holding_days)

        actual_sell_date = actual_sell_price = actual_sell_qty = actual_sell_amount = prft_amt = None
        order_type = ""

        # 매수 조건 성립 + 목표량>0 인 경우에만 매수/청산 로직 진입
        if actual_close and target_price >= actual_close and target_qty > 0:
            buy_qty = target_qty
            buy_amt = round(buy_qty * actual_close, 2)

            # --- 청산 탐색: 이 'buy_qty' 만큼만 매도 ---
            hold_range = market_days[(market_days >= day)][:holding_days]
            future_prices = ticker_data.loc[ticker_data.index.isin(hold_range)]
            match = future_prices[future_prices["Close"] >= sell_target_price]

            if not match.empty:
                actual_sell_date = match.index[0].date()
                actual_sell_price = round(match.iloc[0]["Close"], 2)
            elif moc_sell_date and pd.Timestamp(moc_sell_date) in ticker_data.index:
                actual_sell_date = moc_sell_date
                actual_sell_price = round(ticker_data.loc[pd.Timestamp(moc_sell_date)]["Close"], 2)

            if actual_sell_price:
                actual_sell_qty = buy_qty                 # ← 매수 수량과 동일
                actual_sell_amount = round(actual_sell_price * actual_sell_qty, 2)
                prft_amt = round((actual_sell_amount or 0) - (buy_amt or 0), 2)

            if actual_sell_date:
                order_type = "MOC" if actual_sell_date == moc_sell_date else "LOC"
            elif moc_sell_date == end_dt.date():
                order_type = "MOC"

        else:
            # 당일 미체결(매수 없음)
            actual_close = None
            sell_target_price = None
            moc_sell_date = None
            prft_amt = 0.0

        acum_prft_amt += prft_amt or 0.0

        result.append({
            "일자": day.date(),
            "종가": today_close,
            "모드": mode,
            "변동률": round((today_close - prev_close) / prev_close * 100, 2) if today_close and prev_close else np.nan,
            "매수예정": None,
            "LOC매수목표": target_price,
            "목표량": target_qty if target_qty > 0 else None,
            "매수가": actual_close,
            "매수량": buy_qty if buy_qty > 0 else None,
            "매수금액": buy_amt,
            "매수수수료": None,            
            "매도목표가": sell_target_price,
            "MOC매도일": moc_sell_date,
            "실제매도일": actual_sell_date,
            "실제매도가": actual_sell_price,
            "실제매도량": None,
            "실제매도금액": actual_sell_amount,
            "매도수수료": None,            
            "당일실현": None,                        
            "매매손익": None,
            "누적매매손익": None,
            "복리금액": None,
            "자금갱신": None,            
            "예수금": None,                        
            "주문유형": order_type
        })

        day_cnt += 1
 

    # --- 후처리 ---
    # 1. 1차 루프: 매수예정, 매수량, 매수금액, 실제매도금액, 매매손익, 누적매매손익, 예수금 계산
    prev_pmt_update = first_amt
    prev_cash = first_amt
    prev_profit_sum = 0
        
    for i, row in enumerate(result):
        ### 1. 복리 반영 자금갱신 (1차 루프에서는 미반영)
        자금갱신 = prev_pmt_update
        row["자금갱신"] = round(자금갱신, 2)
        prev_pmt_update = 자금갱신

        ### 2. 매수예정 (직전 자금갱신 기준)
        if i == 0:
            base_amt = round(first_amt / div_cnt, 2)
        else:
            prev_row = result[i - 1]
            prev_update = prev_row.get("자금갱신", 0)
            base_amt = round(prev_update / div_cnt, 2)

        매수예정 = min(base_amt, prev_cash) if prev_cash is not None else None
        row["매수예정"] = 매수예정

        ### 3. 목표량 / 매수량 / 매수금액
        tgt_price = row.get("LOC매수목표")
        buy_price = row.get("매수가")
        sell_price = row.get("실제매도가", 0) or 0

        if 매수예정 and tgt_price:
            qty = int(매수예정 // tgt_price)
        else:
            qty = None

        row["목표량"] = qty
        row["매수량"] = qty
        # ✅ 매수금액을 먼저 계산
        row["매수금액"] = round(qty * buy_price, 2) if qty and buy_price else None

        # ✅ 매수금액이 확정된 후, 실제매도량과 실제매도금액 계산
        if qty and buy_price:
            row["실제매도량"] = qty
            
            if sell_price:
                row["실제매도금액"] = round(row["실제매도량"] * sell_price, 2)
                # ✅ 실제매도금액이 확정된 후 매매손익 계산
                row["매매손익"] = round(row["실제매도금액"] - row["매수금액"], 2)
            else:
                row["실제매도금액"] = None
                row["매매손익"] = None
        else:
            row["실제매도량"] = None
            row["실제매도금액"] = None
            row["매매손익"] = None

        if row["매매손익"] is not None:
            prev_profit_sum += row["매매손익"]
        row["누적매매손익"] = round(prev_profit_sum, 2)

        ### 4. 예수금 = 이전 예수금 - 매수금액 + 당일 매도금액
        buy_amt = row.get("매수금액", 0) or 0
        trade_day = row.get("일자")
        sell_amt = sum(
            s.get("실제매도금액", 0) or 0
            for s in result if s.get("실제매도일") == trade_day
        )
        예수금 = round(prev_cash - buy_amt + sell_amt, 2) if prev_cash is not None else None
        row["예수금"] = 예수금
        prev_cash = 예수금
    
    # 2. 2차 루프: 매매손익이 확정된 후 당일실현 및 복리금액 계산
    for i, row in enumerate(result):
        day = row["일자"]
        row["당일실현"] = round(sum((r.get("매매손익") or 0) for r in result if r.get("실제매도일") == day), 2) or None
        
        if (i + 1) % INVT_RENWL_CYLE == 0:
            bfs = sum((r.get("당일실현") or 0) for r in result[max(0, i - INVT_RENWL_CYLE + 1):i + 1])
            rate = PRFT_CMPND_INT_RT if bfs > 0 else LOSS_CMPND_INT_RT
            row["복리금액"] = round(bfs * rate, 2)
        else:
            row["복리금액"] = None

    # 3. 3차 루프: 복리금액을 자금갱신에 반영하고 나머지 항목 재계산
    prev_pmt_update = first_amt
    prev_cash = first_amt
    prev_profit_sum = 0
        
    # 3차 루프에서도 당일실현을 다시 계산해야 하므로, 당일실현을 계산하는 로직을 추가합니다.
    daily_realized_profits = {}
        
    for i, row in enumerate(result):
        ### 1. 복리 반영 자금갱신
        복리금액 = row.get("복리금액", 0) or 0
        자금갱신 = prev_pmt_update + 복리금액 if 복리금액 is not None else prev_pmt_update
        row["자금갱신"] = round(자금갱신, 2)
        prev_pmt_update = 자금갱신

        ### 2. 매수예정 (직전 자금갱신 기준)
        if i == 0:
            base_amt = round(first_amt / div_cnt, 2)
        else:
            prev_row = result[i - 1]
            prev_update = prev_row.get("자금갱신", 0)
            base_amt = round(prev_update / div_cnt, 2)

        매수예정 = min(base_amt, prev_cash) if prev_cash is not None else None
        row["매수예정"] = 매수예정

        ### 3. 목표량 / 매수량 / 매수금액
        tgt_price = row.get("LOC매수목표")
        buy_price = row.get("매수가")
        sell_price = row.get("실제매도가")
        
        if 매수예정 and tgt_price:
            qty = int(매수예정 // tgt_price)
        else:
            qty = None
        
        row["목표량"] = qty
        # ✅ 수정된 부분: 매수가가 있을 경우에만 매수량을 계산하고 할당
        if buy_price is not None:
            if qty:
                row["매수량"] = qty
                row["매수금액"] = round(qty * buy_price, 2)
            else:
                row["매수량"] = None
                row["매수금액"] = None
        else:
            row["매수량"] = None
            row["매수금액"] = None

        # ✅ 매수량 재계산 후 관련 항목 업데이트
        if sell_price is not None:
            if qty:
                row["실제매도량"] = qty
                row["실제매도금액"] = round(row["실제매도량"] * sell_price, 2)
                row["매매손익"] = round(row["실제매도금액"] - (row["매수금액"] or 0), 2)
            else:
                row["실제매도량"] = None
                row["실제매도금액"] = None
                row["매매손익"] = None
        else:
            row["실제매도량"] = None
            row["실제매도금액"] = None
            row["매매손익"] = None

        ### 4. 누적매매손익
        if row.get("매매손익") is not None:
            prev_profit_sum += row["매매손익"]
        row["누적매매손익"] = round(prev_profit_sum, 2)

        ### 5. 예수금 = 이전 예수금 - 매수금액 + 당일 매도금액
        buy_amt = row.get("매수금액", 0) or 0
        trade_day = row.get("일자")
        sell_amt = sum(
            s.get("실제매도금액", 0) or 0
            for s in result if s.get("실제매도일") == trade_day
        )
        예수금 = round(prev_cash - buy_amt + sell_amt, 2) if prev_cash is not None else None
        row["예수금"] = 예수금
        prev_cash = 예수금

    prev_pmt_update = first_amt        
    # ✅ 3차 루프가 완료된 후 당일실현을 최종적으로 재계산
    for i, row in enumerate(result):
        day = row["일자"]
        if day not in daily_realized_profits:
            daily_realized_profits[day] = sum((r.get("매매손익") or 0) for r in result if r.get("실제매도일") == day)
            
        row["당일실현"] = round(daily_realized_profits[day], 2) or None

        # 복리금액 최종 계산
        if (i + 1) % INVT_RENWL_CYLE == 0:
            bfs = sum((r.get("당일실현") or 0) for r in result[max(0, i - INVT_RENWL_CYLE + 1):i + 1])
            rate = PRFT_CMPND_INT_RT if bfs > 0 else LOSS_CMPND_INT_RT
            row["복리금액"] = round(bfs * rate, 2)
        else:
            row["복리금액"] = None

        복리금액 = row.get("복리금액", 0) or 0
        자금갱신 = prev_pmt_update + 복리금액 if 복리금액 is not None else prev_pmt_update
        row["자금갱신"] = round(자금갱신, 2)
        prev_pmt_update = 자금갱신

        ### 매수예정
        if i == 0:
            base_amt = round(first_amt / div_cnt, 2)
        else:
            prev_row = result[i - 1]
            prev_update = prev_row.get("자금갱신", 0)
            base_amt = round(prev_update / div_cnt, 2)

        매수예정 = min(base_amt, prev_cash) if prev_cash is not None else None
        row["매수예정"] = 매수예정

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
            new_sell_orders.append(Order("매도","MOC", 0.0, sell_moc_order.quantity - buy_order.quantity ))
            buy_order = buy_order._replace(quantity=0)
        elif sell_moc_order.quantity == buy_order.quantity:
            buy_order = buy_order._replace(quantity=0)
        else:
            buy_order = buy_order._replace(quantity=buy_order.quantity - sell_moc_order.quantity)
            if not filtered_sell_orders:
                new_sell_orders.append(Order("매도","LOC", round(buy_order.price + 0.01, 2), sell_moc_order.quantity))

    filtered_sell_orders.sort(key=lambda x: x.price)

    # LOC 매도 주문 퉁치기
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

# ----- 퉁치기 표 색상 지정
def highlight_order(row):
    if row["매매유형"] == "매도":
        return ['background-color: #D9EFFF'] * len(row)  # 하늘색
    elif row["매매유형"] == "매수":
        return ['background-color: #FFE6E6'] * len(row)  # 분홍색
    else:
        return [''] * len(row)
    
# ---------------------------------------
# ✅ Streamlit UI
# ---------------------------------------
st.title("📈 RSI 동적 매매")

# ---------------------------------------
# ✅ 주요 파라미터 입력 (전략 설정값)
# ---------------------------------------
day_cnt = 0 
acum_prft_amt = 0

# ---------------------------------------
# 공통 파라미터
# ---------------------------------------

st.subheader("💹 공통 항목 설정")

# 티커 선택 + 투자금액 입력
col1, col2 = st.columns(2)

with col1:
    target_ticker = st.selectbox('티커 *', ('SOXL', 'KORU', 'TQQQ', 'BITU'))

with col2:
    first_amt = st.number_input("투자금액(USD) *", value=24000, step=500)
    st.markdown(f"**입력한 투자금액:** {first_amt:,}")

# 분할수
div_cnt = st.number_input("분할수 *", value=7, step=1)

# 시작일자 + 종료일자
col3, col4 = st.columns(2)

with col3:
    start_date = st.date_input("투자시작일 *", value=datetime.today() - timedelta(days=20))

with col4:
    end_date = st.date_input("투자종료일 *", value=datetime.today())

# 빈 줄 추가
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------
# 안전모드 파라미터
# ---------------------------------------
st.subheader("💹 안전모드 설정")

safe_hold_days = st.number_input("최대보유일수", value=30, step=1)

col5, col6 = st.columns(2)
with col5:
    safe_buy_threshold  = st.number_input("매수조건이율(%)", min_value=0.0, max_value=100.0, value=3.0, step=0.1, format="%.1f") / 100

with col6:
    safe_sell_threshold = st.number_input("매도조건이율(%)", min_value=0.0, max_value=100.0, value=0.2, step=0.1, format="%.1f") / 100

# 빈 줄 추가
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------
# 공세모드 파라미터
# ---------------------------------------
st.subheader("💹 공세모드 설정")

aggr_hold_days = st.number_input("최대보유일수", value=7, step=1)

col7, col8 = st.columns(2)
with col7:
    aggr_buy_threshold  = st.number_input("매수조건이율(%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1, format="%.1f") / 100

with col8:
    aggr_sell_threshold = st.number_input("매도조건이율(%)", min_value=0.0, max_value=100.0, value=2.5, step=0.1, format="%.1f") / 100

# 빈 줄 추가
st.markdown("<br>", unsafe_allow_html=True)

if st.button("▶ 전략 실행"):
    status_placeholder = st.empty()
    status_placeholder.info("전략 실행 중입니다...")

    df_result = get_mode_and_target_prices(start_date, end_date, target_ticker, first_amt, day_cnt)

    # NaN 및 None 값을 빈 문자열로 대체하여 출력
    #printable_df = df_result.where(pd.notnull(df_result), "")
    printable_df = df_result.replace({None: np.nan})
    #$printable_df = df_result.replace({None: np.nan}, downcast=False)
    printable_df = printable_df.astype(str).replace({"None": "", "nan": ""})

    if printable_df.empty:
        status_placeholder.empty()
        st.warning("데이터가 없습니다. 입력 조건을 확인하세요.")
    else:
        status_placeholder.empty()
        st.success("전략 실행 완료!")       
        
        # total_buy_qty = df_result["매수량"].fillna(0).sum()
        # total_buy_amt = df_result["매수금액"].fillna(0).sum()

        # total_sell_qty = df_result["실제매도량"].fillna(0).sum()
        # total_sell_amt = df_result["실제매도금액"].fillna(0).sum()

        # # 보유량 계산
        # total_qty = int(total_buy_qty - total_sell_qty)

        # # 보유 매수원가
        # holding_cost = total_buy_amt - total_sell_amt

        # # 평균 단가 게산
        # if total_qty > 0:
        #     avg_prc = holding_cost / total_qty
        # else:
        #     avg_prc = 0


        # 1. 매수/매도 데이터 추출 및 통합
        # df_result에서 매수 데이터 추출
        buy_data = df_result[["일자", "매수가", "매수량"]].copy()
        buy_data.columns = ["date", "price", "quantity"]  # 컬럼 이름 통일

        # df_result에서 매도 데이터 추출
        sell_data = df_result[["실제매도일", "실제매도가", "실제매도량"]].copy()
        sell_data.columns = ["date", "price", "quantity"]

        # 매도 데이터에서 '실제매도량'이 NaN인 행은 제거 (매도 기록 없는 경우 제외)
        sell_data = sell_data.dropna(subset=["quantity"])

        # 매도는 보유량을 줄이므로 수량에 음수(-) 부호를 붙임
        sell_data["quantity"] = -sell_data["quantity"]

        # 2. 결합 및 정렬
        # 매수/매도를 하나의 테이블로 합침
        df = pd.concat([buy_data, sell_data], ignore_index=True)

        # date/price/quantity 중 하나라도 비어있는 행 제거
        df = df.dropna(subset=["date", "price", "quantity"])

        # date를 datetime 형식으로 변환 (정렬 및 비교 위해)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # 날짜순으로 정렬
        df = df.sort_values("date").reset_index(drop=True)

        # 3. 평균단가 계산
        avg_prc = 0  # 평균단가 초기화
        history = []  # (날짜, 평균단가) 기록 리스트

        # 데이터에 포함된 고유 날짜를 순서대로 가져옴
        unique_dates = df["date"].unique()

        # 각 날짜별로 처리
        for date in unique_dates:
            # 해당 날짜의 모든 거래 추출
            sub = df[df["date"] == date]
            
            # 첫 거래의 가격 (날짜별 거래는 모두 동일 가격이라는 가정)
            p = sub["price"].iloc[0]
            
            # 해당 날짜의 거래 총수량 (매수는 양수, 매도는 음수)
            q = sub["quantity"].sum()
            
            # 이 날짜 이전까지 누적 보유량
            past_qty = df[df["date"] < date]["quantity"].sum()

            if avg_prc == 0:
                # 첫 매수일에는 평균단가 = 매수가
                avg_prc = p
            elif q < 0:
                # 매도일에는 평균단가 유지 (평단 변동 없음)
                pass
            else:
                # 매수일에는 새 평균단가 계산
                avg_prc = (avg_prc * past_qty + p * q) / (past_qty + q)

            # 이 날짜의 평균단가를 기록
            history.append((date.date(), round(avg_prc, 4)))

        # 4. 출력
        # for h in history:
        #     print(f"{h[0]} → 평균단가: {h[1]}")

        # 5. 최종 결과

        # 현재 보유량 계산 (모든 수량의 합계)
        total_qty = int(df["quantity"].sum())

        #print("\n📌 최종 평균단가:", round(avg_prc, 4))
        #print("📌 최종 보유수량:", total_qty)


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
        
        ## print("-----------total_qty : ", total_qty)
        # 💹 누적매매손익 & 수익률 표시
        # col1, col2, col3, col4 = st.columns(4)
        # col1.metric("📦 보유량", f"{total_qty:,} 주")  
        # col2.metric("💵 평균 단가", f"${avg_prc:,.2f}")
        # col3.metric("📈 누적 매매손익", f"${total_profit:,.2f}")
        # col4.metric("📊 수익률(누적매매손익 / 투자원금)", f"{profit_ratio:.2f} %")

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
            "당일실현": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",                                                            
            "매매손익": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "누적매매손익": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",            
            "복리금액": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",                                                
            "자금갱신": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",                                    
            "예수금": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",                        
        })

        # 빈 줄 추가( 간격 띄우기 위함 )
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 매매 리스트")

        st.dataframe(styled_df)

#        csv = df_result.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

#        st.download_button("⬇️ CSV 다운로드", csv, "strategy_result.csv", "text/csv")

        # 엑셀 데이터 바이너리로 변환
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_result.to_excel(writer, sheet_name="매매리스트", index=False)
        excel_data = output.getvalue()

        # 다운로드 버튼
        st.download_button(
            label="⬇️ 엑셀 다운로드",
            data=excel_data,
            file_name="strategy_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

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
    
