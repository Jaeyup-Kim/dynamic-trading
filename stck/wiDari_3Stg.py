import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================
# 위대리 → 고평가/중립/저평가 단계로 구분
# ============================================

# ============================================
# 1) 일간 → 금요일 기준 주간 데이터 변환
# ============================================

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
    로그 가격 기반 추세선 계산
    """
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
# 3) 메인 리밸런싱 로직
# ============================================

def smart_value_rebalancing_weekly():
    print("=== 📅 스마트 밸류 리밸런싱 (FDR 기반 버전) ===")
    print("📡 데이터를 분석 중입니다...")

    start_date = "2010-01-01"

    # 주간 데이터 변환
    qqq = get_weekly_data_from_daily("QQQ", start_date)
    tqqq = get_weekly_data_from_daily("TQQQ", start_date)

    # 2010-02-12 이후만 사용
    start_filter = "2010-02-12"
    qqq = qqq[qqq.index >= start_filter]
    tqqq = tqqq[tqqq.index >= start_filter]

    # 기준 날짜 (마지막 금요일)
    last_date = qqq.index[-1].strftime("%Y-%m-%d")

    # 추세선 가격
    trend_price = calculate_growth_trend(qqq.values, qqq.index)

    curr_qqq = qqq.iloc[-1]
    curr_tqqq = tqqq.iloc[-1]
    prev_tqqq = tqqq.iloc[-2] if len(tqqq) > 1 else curr_tqqq

    # 이격도
    gap = (curr_qqq - trend_price) / trend_price

    # 상태 판정
    if gap > 0.05:
        status = "🔴 고평가"; sell_rate, buy_rate = 1.00, 0.67
    elif gap < -0.06:
        status = "🔵 저평가"; sell_rate, buy_rate = 0.75, 1.50
    else:
        status = "🟢 중립"; sell_rate, buy_rate = 0.67, 0.67

    print(f"\n📅 기준 날짜: {last_date} (Weekly Close)")
    print("--------------------------------------------------")
    print(f"[1] 시장 진단")
    print(f"  - QQQ 현재가 : ${curr_qqq:.2f}")
    print(f"  - QQQ 적정가 : ${trend_price:.2f} (GROWTH)")
    print(f"  - TQQQ 현재가: ${curr_tqqq:.2f}")
    print(f"  - 이격도     : {gap*100:.2f}% → {status}")
    print("--------------------------------------------------")

    # 자산 입력
    try:
        my_shares = float(input("👉 보유 TQQQ 수량 (주): "))
        my_cash = float(input("👉 매수 가능 현금 (USD): "))
    except:
        print("입력 오류: 숫자만 입력하세요.")
        return

    # 변동 계산
    price_change = curr_tqqq - prev_tqqq
    pnl = price_change * my_shares
    change_pct = (price_change / prev_tqqq) * 100 if prev_tqqq != 0 else 0

    print("\n[2] 변동 분석")
    print(f"  - 주간 TQQQ 변동: ${price_change:.2f} ({change_pct:+.2f}%)")
    print(f"  - 내 평가금 변동: ${pnl:.2f}")
    print("--------------------------------------------------")

    # 매매 계산
    warning = ""   # <-- 기본값 추가
    if pnl > 0:
        # 매도
        action = "매도"
        trade_amount = pnl * sell_rate
        trade_shares = round(trade_amount / curr_tqqq)
    elif pnl < 0:
        # 매수
        action = "매수"
        need = abs(pnl) * buy_rate
        if need > my_cash:
            need = my_cash
            warning = "(보유 현금으로 조정됨)"
        else:
            warning = ""
        trade_shares = round(need / curr_tqqq)

        # 초과 확인
        if trade_shares * curr_tqqq > my_cash:
            trade_shares = int(my_cash // curr_tqqq)
    else:
        action = "없음"
        trade_shares = 0

    # 결과 출력
    print("\n📢 [이번 주 매매 제안]")
    if trade_shares == 0:
        print("   👉 매매 없음 (변동폭 미미)")
    else:
        print(f"   👉 TQQQ {trade_shares}주 {action}")
        print(f"      예상 금액: ${trade_shares * curr_tqqq:.2f}")
        if warning:
            print(f"      ⚠️ {warning}")

    print("--------------------------------------------------")


# ============================================
# 실행
# ============================================

if __name__ == "__main__":
    smart_value_rebalancing_weekly()
