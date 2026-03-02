import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
import sys

# --- [0] 설정 및 경로 정의 ---
FILE_NAME = 'JY_티큐_매매_기록.csv' 
FILE_PATH = os.path.join(os.getcwd(), FILE_NAME) 

# ⭐ CSV 저장 인코딩 설정 (Windows 호환) ⭐
CSV_ENCODING = 'cp949' 

# ⭐ 기본 설정값 (입력 없을 시 사용) ⭐
DEFAULT_INITIAL_CASH = 20000.0
DEFAULT_TARGET_START_DATE = "2010-02-12" 

# --- [1] 데이터 함수 ---

def get_weekly_data_from_daily(ticker, start_date):
    """ 
    FinanceDataReader를 사용하여 일간 데이터 -> 주간 데이터(금요일 기준) 변환
    """
    try:
        df_daily = fdr.DataReader(ticker, start=start_date)
    except Exception as e:
        print(f"❌ {ticker} 데이터 다운로드 실패 (FinanceDataReader): {e}")
        sys.exit(1)

    close_series = df_daily['Close']
    df_weekly = close_series.resample('W-FRI').last().dropna()
    return df_weekly

def calculate_growth_trend(prices, dates):
    """ 
    GROWTH 추세선 계산 (로그 회귀 기반, 최근 260주)
    """
    # 최근 260주(약 5년) 데이터만 사용
    if len(prices) > 260:
        prices = prices[-260:]
        dates = dates[-260:]

    dates_ordinal = np.array([d.toordinal() for d in dates])
    prices_array = np.array(prices)
    
    valid_mask = prices_array > 0
    if np.sum(valid_mask) < 2:
        return prices_array[-1] if np.any(valid_mask) else 0.0
        
    dates_ordinal = dates_ordinal[valid_mask]
    prices_array = prices_array[valid_mask]
    
    log_prices = np.log(prices_array)
    
    slope, intercept = np.polyfit(dates_ordinal, log_prices, 1)
    
    return np.exp(slope * dates_ordinal[-1] + intercept)

# --- [2] 메인 백테스트 로직 함수 ---

def jy_tqqq_backtest(initial_cash, target_start_date):
    print("\n=============================================")
    print(f"=== 💰 JY 티큐 장투매매법 백테스트 시작 ({target_start_date}~) ===")
    print("=============================================")
    print(f"✅ CSV 저장 인코딩: **{CSV_ENCODING}** (Windows 호환)")

    # --- 데이터 로드 ---
    qqq_weekly = get_weekly_data_from_daily('QQQ', target_start_date)
    tqqq_weekly = get_weekly_data_from_daily('TQQQ', target_start_date)

    start_date_dt = pd.to_datetime(target_start_date)
    qqq_weekly = qqq_weekly[qqq_weekly.index >= start_date_dt]
    tqqq_weekly = tqqq_weekly[tqqq_weekly.index >= start_date_dt]
    
    if qqq_weekly.empty or tqqq_weekly.empty or len(qqq_weekly) < 2:
        print("❌ QQQ/TQQQ 데이터 부족. 백테스트를 시작할 수 없습니다.")
        return

    # --- 백테스트 초기화 ---
    current_shares = 0.0
    current_cash = initial_cash
    log_records = []
    data_points = qqq_weekly.shape[0]

    # --- 초기화 기록 (첫 주: 2010-02-12) ---
    first_date = qqq_weekly.index[0].strftime('%Y-%m-%d')
    cash_ratio_init = initial_cash / initial_cash * 100 if initial_cash > 0 else 0.0
    
    first_row = {
        'Date': first_date,
        'TQQQ': round(tqqq_weekly.iloc[0], 2),
        'QQQ': round(qqq_weekly.iloc[0], 2),
        'QQQ성장추세': round(qqq_weekly.iloc[0], 2),
        'Signal': "Initial",
        '매매평가': 0.0,
        '주식잔고': 0.0,
        '평가금': 0.0,
        '매매금': 0.0,
        'Limit': 0.0,      
        'NEW평가금': 0.0,  
        'NEW매매금': 0.0,  
        'NEW매수량': 0.0,  
        'NEW수량': 0.0,    
        'CASH': initial_cash,
        'Total': initial_cash,
        '현금비중': round(cash_ratio_init, 2),
        'MDD': 0.0,       
        'Note': f"초기 투자금 ${initial_cash:,.2f} 설정"
    }
    log_records.append(first_row)
    
    
    # --- 백테스트 반복 (두 번째 주부터 매매 가능) ---
    for i in range(1, data_points):
        
        # 1. 현재 주간 데이터 준비
        current_date = qqq_weekly.index[i]
        curr_qqq_slice = qqq_weekly.iloc[:i+1] 
        curr_qqq = qqq_weekly.iloc[i]
        curr_tqqq = tqqq_weekly.iloc[i] 
        prev_tqqq = tqqq_weekly.iloc[i-1] 

        # 2. 추세선 및 신호 계산
        current_trend = calculate_growth_trend(curr_qqq_slice.values, curr_qqq_slice.index)
        gap = (curr_qqq - current_trend) / current_trend

        status, sell_rate, buy_rate = "Normal (중립)", 0.67, 0.67
        if gap > 0.05:
            status = "Over (고평가)"; sell_rate = 1.00; buy_rate = 0.67
        elif gap < -0.06:
            status = "Under (저평가)"; sell_rate = 0.75; buy_rate = 1.50
            
        # 3. 매매 계산
        p_change = curr_tqqq - prev_tqqq
        pnl = p_change * current_shares 

        action, exec_shares, warning = "Hold", 0.0, ""

        # 가격 상승 시, 보유 주식이 있어야 매도 가능
        if p_change > 0 and current_shares > 0:
            action = "매도"
            amt_to_sell = pnl * sell_rate
            exec_shares_calc = round(amt_to_sell / curr_tqqq)
            exec_shares = min(exec_shares_calc, int(current_shares))
            if exec_shares_calc > int(current_shares): 
                warning = "보유 주식 부족으로 조정"
        # 가격 하락 시, 매수 진행
        elif p_change < 0:
            action = "매수" 
            if current_shares > 0:
                # 기존 보유 주식이 있을 경우, PNL 기반으로 추가 매수량 결정
                amt_to_buy = abs(pnl) * buy_rate
            else:
                # 첫 매수. 보유 주식이 없으므로 초기 자본의 10%를 사용
                amt_to_buy = initial_cash / 10.0
                warning = "첫 매수"
            max_buy_shares = int(current_cash // curr_tqqq)
            suggested_shares = round(amt_to_buy / curr_tqqq)
            exec_shares = min(suggested_shares, max_buy_shares)
            
            if suggested_shares > max_buy_shares and warning != "첫 매수":
                warning = "현금 부족으로 조정"

        # 4. 포트폴리오 업데이트
        exec_price = curr_tqqq 
        
        if action == '매수' and exec_shares > 0:
            cash_flow = -(exec_shares * exec_price)
            current_shares += exec_shares
            current_cash += cash_flow
        elif action == '매도' and exec_shares > 0:
            cash_flow = (exec_shares * exec_price)
            current_shares -= exec_shares
            current_cash += cash_flow
        
        # 5. 최종 자산 계산 및 현금 비중
        total_val = (current_shares * curr_tqqq) + current_cash
        cash_ratio = (current_cash / total_val) * 100 if total_val > 0 else 0.0

        # 6. 기록 저장 (새 양식 적용)
        new_row = {
            'Date': current_date.strftime('%Y-%m-%d'),
            'TQQQ': round(curr_tqqq, 2),
            'QQQ': round(curr_qqq, 2),
            'QQQ성장추세': round(current_trend, 2),
            'Signal': status,
            '매매평가': round(gap*100, 2),
            '주식잔고': round(current_shares, 2),
            '평가금': round(exec_shares * curr_tqqq, 2) if exec_shares > 0 else 0.0,
            '매매금': round(exec_price, 2) if exec_shares > 0 else 0.0,
            'Limit': 0.0,
            'NEW평가금': 0.0,
            'NEW매매금': 0.0,
            'NEW매수량': 0.0,
            'NEW수량': 0.0,
            'CASH': round(current_cash, 2),
            'Total': round(total_val, 2),
            '현금비중': round(cash_ratio, 2),
            'MDD': 0.0,
            'Note': warning if warning else (f"자동 {action} 체결" if exec_shares > 0 else "Hold")
        }
        
        log_records.append(new_row)
        
    # --- 최종 저장 ---
    final_log_df = pd.DataFrame(log_records)
    
    # ⭐ CSV 저장 시 인코딩을 'cp949'로 지정 (Windows 호환) ⭐
    final_log_df.to_csv(FILE_PATH, index=False, encoding=CSV_ENCODING)
    
    # --- 결과 출력 ---
    last_row = final_log_df.iloc[-1]
    
    print("\n=============================================")
    print(f"✅ **백테스트 완료!** (기록 파일: '{FILE_NAME}')")
    print(f"💡 **Windows에서 깨짐 없이 열립니다.** (인코딩: {CSV_ENCODING})")
    print("---------------------------------------------")
    print(f"📈 **최종 총 자산:** **${last_row['Total']:,.2f}**")
    print(f"   - 현금(CASH): ${last_row['CASH']:,.2f}")
    print("=============================================")

if __name__ == "__main__":
    # 사용자 입력 받기
    try:
        input_cash = input(f"👉 초기 투자금 입력 (USD) [기본값: {DEFAULT_INITIAL_CASH}]: ").strip()
        if not input_cash:
            initial_cash = DEFAULT_INITIAL_CASH
        else:
            initial_cash = float(input_cash)
    except ValueError:
        print(f"❌ 잘못된 입력입니다. 기본값 {DEFAULT_INITIAL_CASH}로 설정합니다.")
        initial_cash = DEFAULT_INITIAL_CASH

    input_date = input(f"👉 백테스트 시작일 입력 (YYYY-MM-DD) [기본값: {DEFAULT_TARGET_START_DATE}]: ").strip()
    if not input_date:
        target_start_date = DEFAULT_TARGET_START_DATE
    else:
        target_start_date = input_date

    jy_tqqq_backtest(initial_cash, target_start_date)