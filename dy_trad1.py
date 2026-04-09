import streamlit as st
import pandas as pd
import gspread # Google Sheets 연동 라이브러리
import yfinance as yf
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from collections import namedtuple
import numpy as np
import FinanceDataReader as fdr
import io
import json
import time
import os

# -----------------------------------------------
# 공격형2 포함 : https://dynamic-trading-choice.streamlit.app/
# -----------------------------------------------

# --- 고유 식별자 설정 ---
# 시트의 행을 검색하는 기준이 되는 고유 키 컬럼 이름입니다.
ID_COLUMN_NAME = 'UserID' 
# ---------------------------------------
# ✅ Google Sheets 클라이언트 및 워크시트 초기화
# ---------------------------------------
@st.cache_resource(ttl=3600) # 1시간 동안 연결 정보 캐시
def get_sheets_client():
    """Secrets에서 Google 서비스 계정 정보를 로드하여 GSheets 클라이언트를 반환합니다."""
    # Secrets의 JSON 문자열을 딕셔너리로 변환
    try:
        creds_json = st.secrets["google_service_account_key"]
        if isinstance(creds_json, str):
            creds_dict = json.loads(creds_json)
        else:
            creds_dict = dict(creds_json)
        
        # GSheets 클라이언트 초기화
        client = gspread.service_account_from_dict(creds_dict)
        return client
    except Exception as e:
        st.warning(f"Google Sheets 설정을 찾을 수 없습니다. 로컬 기본값 모드로 실행합니다.\n(상세: {e})")
        return None
        
client = get_sheets_client()

try:
    if client:
        url = st.secrets.get("google_sheet_url")
    else:
        url = None
except Exception:
    # secrets.toml 파일이 없거나 읽을 수 없는 경우
    url = None

@st.cache_resource(ttl=3600)
def get_spreadsheet(_client, url):
    """스프레드시트 객체를 한 번만 열고 캐시합니다. (클라이언트 인수는 해시에서 제외)"""
    if _client is None or not url:
        return None
    try:
        return _client.open_by_url(url)
    except Exception as e:
        st.error(f"Google Sheets 접근 중 오류 발생 (URL 확인 필요): {e}")
        st.stop()

workbook = get_spreadsheet(client, url)

def get_worksheet(sheet_name):
    """지정된 워크시트 이름을 사용하여 워크시트 객체를 반환합니다."""
    if workbook is None:
        return None
    try:
        # 이미 캐시된 workbook 객체를 사용합니다.
        worksheet = workbook.worksheet(sheet_name)
        return worksheet
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"Google Sheet에 '{sheet_name}' 워크시트가 없습니다. 워크시트를 만들어 주세요.")
        st.stop()
    except Exception as e:
        st.error(f"Google Sheets 접근 중 오류 발생 (워크시트: {sheet_name}): {e}")
        st.stop()

# ---------------------------------------
# ✅ 하드코딩된 기본값 및 유틸리티 함수
# ---------------------------------------

# 시트와 무관하게 사용할 기본 파라미터 정의
HARDCODED_DEFAULTS = {
    "style_option": 'Default',
    "target_ticker": 'TQQQ',
    "first_amt": 1000,
    "start_date": '2020-01-01',
}

def get_hardcoded_default_params():
    """시트와 무관하게 코드에 하드코딩된 기본 파라미터를 반환합니다."""
    # 현재 날짜를 end_date 기본값으로 설정
    defaults = HARDCODED_DEFAULTS.copy()
    defaults["end_date"] = datetime.now().strftime('%Y-%m-%d')
    return defaults

# ---------------------------------------
# ✅ 설정 및 파라미터 저장/불러오기 함수 (Google Sheets 기반으로 변경됨)
# ---------------------------------------
def load_user_mappings_from_config(workbook):
    """
    Google Sheets의 'Config' 워크시트에서 'UserID'와 'UserName' 매핑 리스트를 불러옵니다.
    :param workbook: gspread.Spreadsheet 객체
    :return: UserID와 UserName이 매핑된 딕셔너리 리스트
    """
    if not workbook:
        st.error("스프레드시트 객체가 초기화되지 않았습니다. Config 시트를 로드할 수 없습니다.")
        return []

    try:
        # 1. 'Config' 워크시트 객체 가져오기
        config_ws = workbook.worksheet("Config")
        
        # 2. 모든 데이터 읽기
        data = config_ws.get_all_values()
        
        user_mappings = []
        is_user_table = False
        
        # 3. 데이터 파싱
        for row in data:
            # 헤더 행 찾기 ('UserID'와 'UserName'이 A, B열에 있는지 확인)
            if len(row) >= 2 and row[0].strip() == ID_COLUMN_NAME and row[1].strip() == 'UserName':
                is_user_table = True
                continue # 헤더 행은 건너뛰고 다음 행부터 데이터로 처리
            
            # 사용자 데이터 테이블 영역 처리
            if is_user_table:
                # 첫 열(UserID)이 비어있으면 데이터 테이블 끝으로 간주하고 종료
                if not row or not row[0].strip():
                    if not row[0].strip() and not row[1].strip():
                        break
                    continue

                # UserID와 UserName 매핑
                user_id = row[0].strip()
                # B열이 없거나 비어있으면 UserID를 UserName으로 사용
                user_name = row[1].strip() if len(row) > 1 and row[1].strip() else user_id
                
                user_mappings.append({
                    ID_COLUMN_NAME: user_id,
                    'UserName': user_name
                })

        if not user_mappings:
            st.warning("Config 시트에서 'UserID'와 'UserName' 테이블을 찾지 못했거나 데이터가 비어 있습니다. 기본값('default')을 사용합니다.")
            # 데이터가 없을 경우 기본 사용자 ID를 반환
            return [{ID_COLUMN_NAME: "default", "UserName": "기본 사용자"}]
            
        return user_mappings

    except Exception as e:
        st.error(f"Config 시트 사용자 목록 로드 중 오류 발생: {e}. 기본값('default')을 사용합니다.")
        return [{ID_COLUMN_NAME: "default", "UserName": "기본 사용자"}]



def load_params(display_name, unique_id):
    """Google Sheets에서 특정 사용자의 파라미터를 불러옵니다. 없으면 하드코딩된 기본값을 반환합니다."""
    user_params_ws = get_worksheet("UserParams")
    
    # 기본값 가져오기 (사용자 데이터가 없을 경우 반환할 값)
    default_params = get_hardcoded_default_params()

    if user_params_ws is None:
        return default_params

    try:
        data = user_params_ws.get_all_records()
        df = pd.DataFrame(data)
    except Exception as e:
        st.warning(f"'UserParams' 시트 데이터 로드 중 오류 발생: {e}. 하드코딩된 기본값 사용.")
        return default_params

    # 1. 고유 ID(UserID)를 기준으로 해당 사용자의 데이터가 있는지 확인
    # Sheets에서는 ID_COLUMN_NAME을 'UserID'로 사용하고 있습니다.
    user_row = df[df[ID_COLUMN_NAME] == unique_id]
    
    if not user_row.empty:
        # 사용자 데이터가 존재하는 경우
        params_data = user_row.iloc[0]
        # 데이터가 있으면 해당 사용자의 파라미터를 반환
        return {
            "style_option": str(params_data.get('style_option', default_params['style_option'])),
            "target_ticker": str(params_data.get('target_ticker', default_params['target_ticker'])),
            # 값이 없을 경우 기본값 사용 (int() 변환 시 오류 방지)
            "first_amt": int(params_data.get('first_amt', default_params['first_amt'])),
            "start_date": str(params_data.get('start_date', default_params['start_date'])),
            # end_date는 시트에서 값을 가져오지 않고 현재 날짜(default_params에서 가져옴)를 기본값으로 사용
            "end_date": default_params['end_date']
        }
    else:
        # 2. 사용자 데이터가 없으면 하드코딩된 기본값을 반환 (시트 접근 없음)
        st.info(f"사용자 '{display_name}' ({unique_id})의 파라미터가 시트에 없습니다. 기본 설정으로 시작합니다.")
        return default_params    

def save_params_robust(params, unique_id, display_name):
    """파라미터를 Google Sheets의 'UserParams'에 고유 ID(UserID)를 기준으로 저장하거나 업데이트합니다."""
    try:
        # 1. 시트 연결 및 데이터 준비
        user_params_ws = get_worksheet("UserParams")
        
        if user_params_ws is None:
            st.warning("Google Sheets에 연결되어 있지 않아 설정을 저장할 수 없습니다.")
            return
        
        # 시트 헤더를 가져와서 업데이트할 값의 순서를 맞춥니다.
        headers = user_params_ws.row_values(1)
        
        # 고유 ID 컬럼의 위치를 찾습니다.
        if ID_COLUMN_NAME not in headers:
            raise ValueError(f"시트 헤더에 필수 컬럼 '{ID_COLUMN_NAME}'을(를) 찾을 수 없습니다.")
        
        # 저장할 데이터 딕셔너리 준비 (UserID와 User(이름) 모두 포함)
        data_to_save = {
            ID_COLUMN_NAME: unique_id,           # 🔑 고유 ID (검색 키)
            'UserName': display_name,            # 📝 변경 가능한 사용자 이름
            'style_option': params.get('style_option', ''),
            'target_ticker': params.get('target_ticker', ''),
            'first_amt': params.get('first_amt', ''),
            'start_date': params.get('start_date', ''),
            'end_date': '' # end_date는 저장하지 않음
        }
        
        # 데이터 목록 준비 (시트 헤더 순서에 맞춤)
        row_values = [data_to_save.get(h, '') for h in headers] 

        # 2. 고유 ID를 기반으로 행 찾기 (Upsert 로직 시작)
        id_column_index = headers.index(ID_COLUMN_NAME) + 1 # gspread는 1-based 인덱스 사용
        
        # 'UserID' 열의 모든 값을 가져옵니다. (효율적인 검색)
        id_column_values = user_params_ws.col_values(id_column_index)
        
        try:
            # id_column_values[1:] : 헤더 제외한 실제 데이터만 검색
            id_data_list = id_column_values[1:] 
            
            # 고유 ID가 존재하는지 확인합니다.
            unique_id_index_in_data = id_data_list.index(unique_id)
            
            # 실제 시트의 행 번호 (1-based, 헤더 1행 + 데이터 시작 1행 + 인덱스 값)
            row_num = unique_id_index_in_data + 2 
            
            # 3. 갱신 (Update)
            # A{row_num} 셀부터 시작하여 row_values의 길이만큼 행을 업데이트합니다.
            update_range = f'A{row_num}'
            user_params_ws.update(range_name=update_range, values=[row_values])
            st.toast(f"✅ 파라미터가 Google Sheets에 업데이트되었습니다. (ID: {unique_id}, 행: {row_num})")
            
        except ValueError:
            # 4. 추가 (Insert): 리스트에 해당 고유 ID가 없는 경우 (ValueError 발생)
            user_params_ws.append_row(row_values)
            st.toast(f"✅ 새 파라미터가 Google Sheets에 저장되었습니다. (ID: {unique_id}, 이름: {display_name})")
            
    except Exception as e:
        st.error(f"Google Sheets 저장 중 오류 발생: {e}")

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

# 투자금 갱신주기 설정
INVT_RENWL_CYLE = 10
# 주문 정보 구조 정의
Order = namedtuple('Order', ['side', 'type', 'price', 'quantity'])

# ============================================
# 최적화 1: 주차 계산 함수 교체
# ============================================
def get_weeknum_google_style(date):
    """
    절대 주차 계산 (년도와 무관하게 연속적인 주차 번호 반환)
    기준일: 2000-01-02 (일요일)
    """
    base_date = pd.Timestamp("2000-01-02")

    if isinstance(date, pd.Series):
        d = pd.to_datetime(date)
        if d.dt.tz is not None:
            d = d.dt.tz_localize(None)
        return (d - base_date).dt.days // 7
    elif isinstance(date, pd.DatetimeIndex):
        if date.tz is not None:
            date = date.tz_localize(None)
        return (date - base_date).days // 7
    else:
        ts = pd.Timestamp(date)
        return (ts.tz_localize(None) - base_date).days // 7

# ============================================
# 최적화 2: 주간 마지막 거래일 추출
# ============================================
def get_last_trading_day_each_week(data):

    """
    최적화된 주간 마지막 거래일 추출 - 절대 주차 사용
    """
    data = data.copy()
    
    # 절대 주차 계산
    data['week'] = get_weeknum_google_style(data.index)
    data['weekday'] = data.index.weekday
    
    # groupby 최적화 (절대 주차 사용하므로 year 그룹핑 불필요)
    last_day = data.groupby(['week'])['weekday'].idxmax()
    return data.loc[last_day]


# ============================================
# 최적화 3: RSI 계산 함수
# ============================================
def calculate_rsi_rolling(data, period=14):

    """
    RSI(상대강도지수)를 주어진 기간 기준으로 계산
    기본: 14일
    """    

    # data = data.copy()
    # delta = data['Close'].diff()
    # gain = delta.where(delta > 0, 0.0)
    # loss = -delta.where(delta < 0, 0.0)
    
    # # rolling 대신 ewm 사용 (더 빠름)
    # avg_gain = gain.rolling(window=period, min_periods=period).mean()
    # avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # # 0으로 나누기 방지
    # rs = avg_gain / avg_loss.replace(0, np.nan)
    # rsi = 100 - (100 / (1 + rs))
    
    # data['RSI'] = rsi.round(2)

    # #print("-----------data : \n", data)
    # return data

    data = data.copy()
    data['delta'] = data['Close'].diff()
    data['gain'] = data['delta'].where(data['delta'] > 0, 0.0)
    data['loss'] = -data['delta'].where(data['delta'] < 0, 0.0)
    data['avg_gain'] = data['gain'].rolling(window=period).mean()
    data['avg_loss'] = data['loss'].rolling(window=period).mean()
    data['RS'] = (data['avg_gain'] / data['avg_loss']).round(3)
    data['RSI'] = ((data['RS'] / (1 + data['RS'])) * 100).round(2)
    
    return data

# ============================================
# 최적화 4: 모드 판별 함수 (벡터화)
# ============================================
def assign_mode_v2(rsi_series):
    """
    최적화된 모드 판별 - 기존 함수 교체용
    """
    mode = pd.Series('방어', index=range(len(rsi_series)))
    
    # 배열로 변환하여 빠른 접근
    rsi_arr = rsi_series.values if hasattr(rsi_series, 'values') else rsi_series
    
    for i in range(2, len(rsi_arr)):
        two_weeks_ago = rsi_arr[i - 2]
        one_week_ago = rsi_arr[i - 1]
        
        ##print("------ two : ", two_weeks_ago, "------- one : ", one_week_ago)

        # 방어 조건
        if (
            (two_weeks_ago > 65 and two_weeks_ago > one_week_ago) or
            (40 < two_weeks_ago < 50 and two_weeks_ago > one_week_ago) or
            (one_week_ago < 50 and 50 < two_weeks_ago)
        ):
            mode.iloc[i] = "방어"
        # 공격 조건
        elif (
            (two_weeks_ago < 35 and two_weeks_ago < one_week_ago) or
            (50 < two_weeks_ago < 60 and two_weeks_ago < one_week_ago) or
            (one_week_ago > 50 and 50 > two_weeks_ago)
        ):
            mode.iloc[i] = "공격"
        else:
            mode.iloc[i] = mode.iloc[i - 1]
    
    return mode.tolist()


def get_future_market_day(start_day, market_days, offset_days):
    """
    기준일로부터 N일 후의 거래일 반환
    예: MOC 매도를 위한 MOC매도일자 계산
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
                sell_orders.append(Order("매도", "LOC", price, qty))
        
        # 실제매도일이 미입력이고 MOC매도일이 존재하고 주문유형이 MOC일 경우
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

    # 매수금액이 아닌 매수예정 기준으로 차감
    planned_buy = row.get("매수예정", 0) or 0
    trade_day = row.get("일자")

    today_sell_profit = sum(
        s.get("실제매도금액", 0)
        for s in sell_list
        if s.get("실제매도일") == trade_day
    )

    return round(prev_balance - planned_buy + today_sell_profit, 2)
 
# ============================================
# 최적화 5: 메인 전략 함수 (핵심 최적화)
# ============================================
@st.cache_data(ttl=1800, show_spinner=False)  # 30분 캐시
def get_mode_and_target_prices(start_date, end_date, target_ticker, first_amt, day_cnt, 
                                dfns_hold_days, dfns_buy_threshold, dfns_sell_threshold, dfns_div_cnt, 
                                atck_hold_days, atck_buy_threshold, atck_sell_threshold, atck_div_cnt, 
                                prft_cmpnd_int_rt, loss_cmpnd_int_rt):

    v_first_amt = first_amt
    result_rows = []

    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)

    yf_end_dt = (end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    qqq_start = start_dt - pd.Timedelta(weeks=25) # RSI 계산을 위한 25주치 데이터 필요

    # 거래일 캘린더
    nyse = mcal.get_calendar("NYSE")
    market_days = nyse.schedule(
        start_date=qqq_start.strftime("%Y-%m-%d"),
        end_date=(end_dt + pd.Timedelta(days=dfns_hold_days + 60)).strftime("%Y-%m-%d")
    ).index.normalize()
    
    # QQQ 데이터 로드
    ##qqq = fdr.DataReader("QQQ", qqq_start.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

    # RSI 계산을 위한 QQQ 데이터 로드 (yfinance 사용)
    qqq = yf.download("QQQ", start=qqq_start.strftime("%Y-%m-%d"), end=end_dt, auto_adjust=False, back_adjust=False, progress=False)

    # MultiIndex 대응 (yfinance 최신 버전 이슈 방지)
    if isinstance(qqq.columns, pd.MultiIndex):
        qqq.columns = qqq.columns.get_level_values(0)

    qqq.index = pd.to_datetime(qqq.index)
    qqq['Close'] = qqq['Close'].round(2)

    if end_dt not in qqq.index: # 종료일자가 데이터에 없으면 추가
        qqq.loc[end_dt] = None

    # 주간 RSI 계산 (최적화)
    weekly = get_last_trading_day_each_week(qqq)
    ###print("---------- weekly : \n", weekly)    
    weekly_rsi = calculate_rsi_rolling(weekly).dropna(subset=["RSI"])

    ##print("---------- weekly_rsi : \n", weekly_rsi)

    weekly_rsi["모드"] = assign_mode_v2(weekly_rsi["RSI"])
    weekly_rsi["week"] = get_weeknum_google_style(weekly_rsi.index)
    mode_by_year_week = weekly_rsi.set_index("week")[["모드", "RSI"]]

    # 타겟 티커 데이터 로드
    ###ticker_data = fdr.DataReader(target_ticker, qqq_start.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    # 타겟 티커 데이터 로드 (yfinance 사용) ---
    ticker_data = yf.download(target_ticker, start=qqq_start.strftime("%Y-%m-%d"), end=end_dt, auto_adjust=False, back_adjust=False,progress=False)    
    
    # MultiIndex 대응
    if isinstance(ticker_data.columns, pd.MultiIndex):
        ticker_data.columns = ticker_data.columns.get_level_values(0)    

    ticker_data.index = pd.to_datetime(ticker_data.index)
    ticker_data['Close'] = ticker_data['Close'].round(2)

    for day in market_days:
        if not (start_dt <= day <= end_dt):
            continue

        week = get_weeknum_google_style(day)
        if week not in mode_by_year_week.index:
            continue

        # 해당 날짜의 연도 및 주차 정보로 모드(RSI 기반) 조회
        mode_info = mode_by_year_week.loc[week]
        if isinstance(mode_info, pd.DataFrame):
            mode_info = mode_info.iloc[0]

        mode = mode_info["모드"]
        rsi = round(mode_info["RSI"], 2)

        prev_days = ticker_data.index[ticker_data.index < day]

        if len(prev_days) == 0:
            continue
        
        prev_close_val = ticker_data.loc[prev_days[-1], "Close"]
        if isinstance(prev_close_val, pd.Series):
            prev_close = round(float(prev_close_val.iloc[0]), 2)
        else:
            prev_close = round(float(prev_close_val), 2)

        # 해당일 종가 (체결 여부 판단용)
        if day in ticker_data.index:
            close_value = ticker_data.loc[day, "Close"]
            # Handle case where indexing returns a Series instead of scalar
            if isinstance(close_value, pd.Series):
                actual_close = close_value.iloc[0] if len(close_value) > 0 else None
            else:
                actual_close = close_value
        else:
            actual_close = None

        if actual_close is not None and pd.notna(actual_close):
            actual_close = round(float(actual_close), 2)
        else:
            actual_close = None

        today_close = actual_close

        if mode == "방어":
            # 모드에 따라 목표가 및 보유일 설정
            div_cnt = dfns_div_cnt
            target_price = round(prev_close * (1 + dfns_buy_threshold), 2)
            sell_target_price = round((actual_close or target_price) * (1 + dfns_sell_threshold), 2)
            holding_days = dfns_hold_days
        else:
            div_cnt = atck_div_cnt
            target_price = round(prev_close * (1 + atck_buy_threshold), 2)
            sell_target_price = round((actual_close or target_price) * (1 + atck_sell_threshold), 2)
            holding_days = atck_hold_days

        # 1회 매수에 사용할 금액 및 목표 수량 계산
        daily_buy_amount = round(v_first_amt / div_cnt, 2)
        ###target_qty = int(daily_buy_amount // target_price) if target_price else 0
        ##target_price_safe = float(target_price) if target_price is not None and pd.notna(target_price) else 0.0

        if isinstance(target_price, pd.Series):
            target_price_safe = float(target_price.iloc[0]) if len(target_price) > 0 and pd.notna(target_price.iloc[0]) else 0
        else:
            target_price_safe = float(target_price) if target_price is not None else 0

        # 2. 가격이 0보다 크고 유효한 값일 경우에만 수량을 계산합니다.
        if target_price_safe > 0:
            # 3. 일일 매수 금액을 방어한 가격으로 나누어 수량을 계산합니다.
            #    일반 나누기(/)를 사용하고 int()로 정수 변환하여 소수점을 버립니다.
            target_qty = int(daily_buy_amount / target_price_safe)
        else:
            # 4. 가격이 0, None, 또는 NaN이면 수량은 0입니다.
            target_qty = 0        

        buy_qty = 0
        buy_amt = None
        moc_sell_date = get_future_market_day(day, market_days, holding_days)
        
        # 초기화: 실제 매도 관련 정보
        actual_sell_date = actual_sell_price = actual_sell_qty = actual_sell_amount = prft_amt = None
        order_type = ""

        # 실제 체결 가능한 경우 (매수 목표가 ≥ 종가)
        if actual_close and target_price >= actual_close and target_qty > 0:
            buy_qty = target_qty
            buy_amt = round(buy_qty * actual_close, 2)
            # 보유 기간 내 종가가 매도 목표가를 넘긴 경우 매도 성사
            hold_range = market_days[(market_days >= day)][:holding_days]
            future_prices = ticker_data.loc[ticker_data.index.isin(hold_range)]
            match = future_prices[future_prices["Close"] >= sell_target_price]

            if not match.empty:
                actual_sell_date = match.index[0].date()
                val = match.iloc[0]["Close"]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                actual_sell_price = round(float(val), 2)
            elif moc_sell_date and pd.Timestamp(moc_sell_date) in ticker_data.index:
                # 조건 달성 실패 시 MOC 매도
                actual_sell_date = moc_sell_date
                val = ticker_data.loc[pd.Timestamp(moc_sell_date)]["Close"]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                actual_sell_price = round(float(val), 2)

            # if actual_sell_date:
            #     if actual_sell_date == moc_sell_date:
            #         order_type = "MOC"
            #     else:
            #         order_type = "LOC"
            # elif moc_sell_date: # actual_sell_date가 None이지만 moc_sell_date가 존재하면 MOC 매도를 의도한 것으로 간주
            #     order_type = "MOC"
            # else:
            #     order_type = "" # 실제 매도일도, MOC 예정일도 없는 경우

            if actual_sell_date:
                if actual_sell_date == moc_sell_date: # 실제매도일이 존재할 경우 실제매도일과 MOC매도일자가 같은 일자이면 MOC 매도
                    order_type = "MOC"
                else:
                    order_type = "LOC"    # 실제매도일과 MOC매도일자가 다르면 LOC 매도        
            elif end_date == moc_sell_date: 
                order_type = "MOC"       # 아직 매도가 되지 않아서 실제매도일이 없을 경우에  end_date 와 moc_sell_date가 같은 일자이면 MOC 매도
            else:
                order_type = "LOC"     # 실제매도일도 없고  end_date 와 moc_sell_date가 다른 일자이면 LOC 매도           

            #print("------- end_date, actual_sell_date, moc_sell_date : ", end_date, actual_sell_date, moc_sell_date)
            #print("-------- order_type : ", order_type)

        else: # 매수 미체결 시: 관련 값 모두 초기화
            actual_close = None
            sell_target_price = None
            moc_sell_date = None
            prft_amt = 0.0

        # 결과 누적
        result_rows.append({
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
           # "매수수수료": None,
            "매도목표가": sell_target_price,
            "MOC매도일": moc_sell_date,
            "실제매도일": actual_sell_date,
            "실제매도가": actual_sell_price,
            "실제매도량": None,
            "실제매도금액": None,
           # "매도수수료": None,
            "당일실현": None,
            "매매손익": None,
            "누적매매손익": None,
            "복리금액": None,
            "자금갱신": None,
            "예수금": None,
            "주문유형": order_type
        })
        day_cnt += 1

    result = pd.DataFrame(result_rows)
    if result.empty:
        return result

    prev_cash = prev_pmt_update = first_amt
    prev_profit_sum = 0.0
    daily_realized_profits = {}

    #print("----------------result : ", result)

    num_cols = ["실제매도금액", "매매손익", "당일실현"]
    for col in num_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    ##for i, row in enumerate(result):
    # result는 이미 pd.DataFrame(result_rows)로 생성되어 있다고 가정
    # prev_cash, prev_pmt_update, first_amt, prev_profit_sum, daily_realized_profits, dfns_div_cnt, atck_div_cnt, INVT_RENWL_CYLE, prft_cmpnd_int_rt, loss_cmpnd_int_rt 등은 기존값 유지

    for i, idx in enumerate(result.index):
        # 행(읽기 전용) 가져오기
        row = result.loc[idx]

        # 모드에 따라 분할수 결정
        if row["모드"] == "방어":
            div_cnt = dfns_div_cnt
        else:
            div_cnt = atck_div_cnt

        # 매수예정(금액) 계산
        base_amt = round((prev_pmt_update if i > 0 else first_amt) / div_cnt, 2)
        buy_plan = base_amt if prev_cash is None else min(base_amt, prev_cash)
        result.loc[idx, "매수예정"] = buy_plan

        # 가격/수량 계산
        tgt_price = row.get("LOC매수목표")
        buy_price = row.get("매수가")
        sell_price = row.get("실제매도가")

        ##qty = int(buy_plan // tgt_price) if (tgt_price and tgt_price > 0) else None


        if isinstance(tgt_price, pd.Series):
            tgt_price_val = float(tgt_price.iloc[0]) if len(tgt_price) > 0 and pd.notna(tgt_price.iloc[0]) else None
        else:
            tgt_price_val = float(tgt_price) if pd.notna(tgt_price) else None

        if isinstance(buy_price, pd.Series):
            buy_price = float(buy_price.iloc[0]) if not buy_price.empty else None

        if isinstance(sell_price, pd.Series):
            sell_price = float(sell_price.iloc[0]) if not sell_price.empty else None

        qty = int(buy_plan // tgt_price_val) if tgt_price_val and tgt_price_val > 0 else None

        result.loc[idx, "목표량"] = qty
        result.loc[idx, "매수량"] = qty if buy_price else None
        result.loc[idx, "매수금액"] = round(qty * buy_price, 2) if (qty and buy_price) else None

        # 매도 처리(실제매도가가 있으면 매매손익 산정)
        if qty and sell_price:
            real_sell_amt = round(qty * sell_price, 2)
            result.loc[idx, "실제매도량"] = qty
            result.loc[idx, "실제매도금액"] = real_sell_amt
            result.loc[idx, "매매손익"] = real_sell_amt - (result.loc[idx, "매수금액"] or 0)
        else:
            result.loc[idx, "실제매도량"] = None
            result.loc[idx, "실제매도금액"] = None
            result.loc[idx, "매매손익"] = None

        # 누적매매손익 업데이트
        if result.loc[idx, "매매손익"] is not None:
            prev_profit_sum += result.loc[idx, "매매손익"]
        result.loc[idx, "누적매매손익"] = prev_profit_sum

        # 동일 거래일의 총 실현(매도)금액 계산 (마스크 사용)
        trade_day = row.get("일자")
        if pd.isna(trade_day):
            sell_amt = 0
        else:
            mask_same_day = result["실제매도일"] == trade_day
            sell_amt = result.loc[mask_same_day, "실제매도금액"].fillna(0).sum()

       # 예수금 업데이트
        # buy_amt = result.loc[idx, "매수금액"] or 0
        # prev_cash = prev_cash - buy_amt + sell_amt
        # result.loc[idx, "예수금"] = prev_cash if row.get("종가") is not None else None

        buy_amt = result.loc[idx, "매수금액"] or 0
        if result.loc[idx, "매수가"] is not None and buy_amt > 0:
            prev_cash -= buy_amt  # 실제 체결시에만 예수금 차감

        sell_amt = result.loc[mask_same_day, "실제매도금액"].fillna(0).sum()
        if not pd.isna(sell_amt) and sell_amt > 0:
            prev_cash += sell_amt
        result.loc[idx, "예수금"] = prev_cash if row.get("종가") is not None else None

        # 당일 실현 손익 집계 (캐시 dict 대신 DataFrame으로 계산 가능)
        # 여기서는 daily_realized_profits dict를 유지하되 key는 trade_day로 통일
        if trade_day not in daily_realized_profits:
            mask = result["실제매도일"] == trade_day
            daily_realized_profits[trade_day] = result.loc[mask, "매매손익"].fillna(0).sum()
        result.loc[idx, "당일실현"] = daily_realized_profits.get(trade_day) or None

        # 복리금액 계산: 최근 INVT_RENWL_CYLE 행의 '당일실현' 합계 사용
        if (i + 1) % INVT_RENWL_CYLE == 0:
            start_pos = max(0, i - INVT_RENWL_CYLE + 1)
            window = result.iloc[start_pos:i + 1]
            bfs = window["당일실현"].fillna(0).sum()
            rate = prft_cmpnd_int_rt if bfs > 0 else loss_cmpnd_int_rt
            result.loc[idx, "복리금액"] = round(bfs * rate, 2)
        else:
            result.loc[idx, "복리금액"] = None

        # 자금갱신 업데이트
        prev_pmt_update += result.loc[idx, "복리금액"] or 0
        result.loc[idx, "자금갱신"] = prev_pmt_update

    # 함수 최종 반환 시에는 이미 DataFrame이므로 그대로 반환
    return result


# ----------상계 처리 표 출력 ----------
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

    return df

def print_orders(sell_orders, buy_orders):
    """
    매도/매수 주문을 구분 출력
    - 매도는 가격 내림차순
    - 매수는 가격 오름차순
    """
    # 이 함수는 콘솔 디버깅용이므로 출력 생략
    pass

# ============================================
# 최적화 6: 상계 처리 함수
# ============================================
def remove_duplicates(sell_orders, buy_orders):
    """
    최적화된 상계 처리 - 기존 함수 교체용
    알고리즘 복잡도 개선: O(n²) -> O(n)
    """
    if not sell_orders or not buy_orders:
        return

    buy_order = buy_orders[0]
    new_sell_orders = []
    new_buy_orders = []
    
    remaining_buy_qty = buy_order.quantity
    
    # MOC 매도 분리
    sell_moc = [o for o in sell_orders if o.type == "MOC"]
    sell_loc = [o for o in sell_orders if o.type == "LOC"]
    
    # LOC 중 상계 대상과 비대상 분리
    sell_loc_match = [o for o in sell_loc if o.price <= buy_order.price]
    sell_loc_nomatch = [o for o in sell_loc if o.price > buy_order.price]
    
    # 1. MOC 상계
    for moc_order in sell_moc:
        if remaining_buy_qty == 0:
            new_sell_orders.append(moc_order)
            continue
        
        matched = min(moc_order.quantity, remaining_buy_qty)
        remaining_buy_qty -= matched
        
        if moc_order.quantity > matched:
            new_sell_orders.append(
                Order("매도", "MOC", 0.0, moc_order.quantity - matched)
            )
    
    # 2. LOC 상계 (가격 낮은 순)
    sell_loc_match.sort(key=lambda x: x.price)
    
    for loc_order in sell_loc_match:
        if remaining_buy_qty == 0:
            new_sell_orders.append(loc_order)
            continue
        
        matched = min(loc_order.quantity, remaining_buy_qty)
        
        new_buy_orders.append(
            Order("매수", "LOC", round(loc_order.price - 0.01, 2), matched)
        )
        
        remaining_buy_qty -= matched
        
        if loc_order.quantity > matched:
            new_sell_orders.append(
                Order("매도", "LOC", loc_order.price, loc_order.quantity - matched)
            )
    
    # 3. 남은 매수 처리
    if remaining_buy_qty > 0:
        new_buy_orders.append(
            Order("매수", "LOC", buy_order.price, remaining_buy_qty)
        )
        
        if sell_loc_match:
            total_matched = buy_order.quantity - remaining_buy_qty
            new_sell_orders.append(
                Order("매도", "LOC", round(buy_order.price + 0.01, 2), total_matched)
            )
    else:
        new_sell_orders.append(
            Order("매도", "LOC", round(buy_order.price + 0.01, 2), buy_order.quantity)
        )
    
    # 4. 비상계 LOC 추가
    new_sell_orders.extend(sell_loc_nomatch)
    
    # 정렬
    new_sell_orders.sort(key=lambda x: x.price, reverse=True)
    new_buy_orders.sort(key=lambda x: x.price, reverse=True)
    
    sell_orders[:] = new_sell_orders
    buy_orders[:] = new_buy_orders

def highlight_order(row):
    """당일주문 리스트색상 지정"""
    if row["매매유형"] == "매도":
        return ['background-color: #D9EFFF'] * len(row)  # 하늘색
    elif row["매매유형"] == "매수":
        return ['background-color: #FFE6E6'] * len(row)  # 분홍색
    else:
        return [''] * len(row)

# ---------------------------------------
# ✅ Streamlit UI
# ---------------------------------------
st.title("📈 RSI 변동성 매매")
# ---------------------------------------
# ✅ 설정 로드 (사용자 이름)
# ---------------------------------------
# --- 1단계: Config 시트에서 사용자 목록 로드 ---
user_mappings = load_user_mappings_from_config(workbook) 

# --- 2단계: UI 구성을 위한 데이터 준비 ---
# 표시 이름 리스트 생성
display_names = [mapping['UserName'] for mapping in user_mappings] 

# 이름(키)으로 ID(값)를 찾기 위한 매핑 딕셔너리 생성
user_id_map = {mapping['UserName']: mapping[ID_COLUMN_NAME] for mapping in user_mappings}

# ---------------------------------------
# ✅ 사이드바에 사용자 이름 관리 섹션 제거 (테이블 기반 관리로 대체)
# ---------------------------------------
##.sidebar.markdown("---")
##.sidebar.info("사용자 목록은 Google Sheets 'Config' 시트의 'UserID'/'UserName' 테이블을 통해 관리됩니다.")
##.sidebar.markdown("---")

# ---------------------------------------
# ✅ 사용자 선택 드롭다운 (고유 ID 추출 로직 포함)
# ---------------------------------------
st.subheader("👨‍💻 사용자 설정")

# 초기 선택값 설정
if 'selected_user_name' not in st.session_state:
    st.session_state.selected_user_name = display_names[0] if display_names else "기본 사용자"

# 현재 목록에 없는 세션 값은 첫 번째 값으로 초기화 (시트에서 목록이 바뀐 경우)
if st.session_state.selected_user_name not in display_names:
    st.session_state.selected_user_name = display_names[0] if display_names else "기본 사용자"

try:
    current_index = display_names.index(st.session_state.selected_user_name)
except ValueError:
    current_index = 0

selected_user_name = st.selectbox("사용자", display_names, index=current_index, label_visibility="collapsed")

# 선택된 사용자 이름과 고유 ID 정의
CURRENT_DISPLAY_NAME = selected_user_name
UNIQUE_ID_KEY = user_id_map.get(CURRENT_DISPLAY_NAME)

if UNIQUE_ID_KEY is None:
    # 이 오류는 Config 시트에 문제가 있을 때 발생합니다.
    st.error("오류: 선택된 사용자에 대한 고유 ID(UserID)를 Config 시트에서 찾을 수 없습니다.")
    st.stop()

if selected_user_name != st.session_state.selected_user_name:
    st.session_state.selected_user_name = selected_user_name
    st.rerun()

# 선택된 사용자의 파라미터 로드 (UserID 기준으로 로드)
# 이제 시트에 사용자 ID가 없으면 하드코딩된 기본값이 로드됩니다.
params = load_params(CURRENT_DISPLAY_NAME, UNIQUE_ID_KEY)

# ---------------------------------------
# 스타일 설정 사전
# ---------------------------------------
styles = {
    "Default": {
        "dfns_hold_days": 30,
        "dfns_buy_threshold": 3.0,
        "dfns_sell_threshold": 0.2,
        "dfns_div_cnt": 7,
        "atck_hold_days": 7,
        "atck_buy_threshold": 5.0,
        "atck_sell_threshold": 2.5,
        "atck_div_cnt": 7,
        "prft_cmpnd_int_rt": 0.8,   # 이익복리율
        "loss_cmpnd_int_rt": 0.3,   # 손실복리율
    },
    "공격형2": {
        "dfns_hold_days": 35,
        "dfns_buy_threshold": 3.5,
        "dfns_sell_threshold": 1.8,
        "dfns_div_cnt": 7,
        "atck_hold_days": 7,
        "atck_buy_threshold": 3.6,
        "atck_sell_threshold": 5.6,
        "atck_div_cnt": 8,
        "prft_cmpnd_int_rt": 0.72,  # 이익복리율
        "loss_cmpnd_int_rt": 0.213, # 손실복리율
    }
}
# ---------------------------------------
# 공통 파라미터
# ---------------------------------------
st.subheader("💹 공통 항목 설정")
# 📝 스타일 선택
style_options = list(styles.keys())
current_style_index = style_options.index(params["style_option"]) if params["style_option"] in style_options else 0
style_option = st.selectbox("스타일 선택", style_options, index=current_style_index)
selected_style = styles[style_option]

if style_option != params["style_option"]:
    params["style_option"] = style_option
    # 고유 ID와 표시 이름 모두 전달
    save_params_robust(params, UNIQUE_ID_KEY, CURRENT_DISPLAY_NAME)

col1, col2 = st.columns(2)

with col1:
    # 📝 티커 선택
    tickers = ('SOXL', 'KORU', 'TQQQ', 'BITU')
    current_ticker_index = tickers.index(params["target_ticker"]) if params["target_ticker"] in tickers else 0
    target_ticker = st.selectbox('티커 *', tickers, index=current_ticker_index)
    
    if target_ticker != params["target_ticker"]:
        params["target_ticker"] = target_ticker
        # 고유 ID와 표시 이름 모두 전달
        save_params_robust(params, UNIQUE_ID_KEY, CURRENT_DISPLAY_NAME)

with col2:
    # 📝 투자금액 입력
    first_amt = st.number_input("투자금액(USD) *", value=params["first_amt"], step=500, min_value=100)
    if first_amt != params["first_amt"]:
        params["first_amt"] = first_amt
        # 고유 ID와 표시 이름 모두 전달
        save_params_robust(params, UNIQUE_ID_KEY, CURRENT_DISPLAY_NAME)
    st.markdown(f"**현재 설정된 투자금액:** {first_amt:,} USD")

# 시작일자 + 종료일자
col3, col4 = st.columns(2)

with col3:
    # 📝 투자 시작일 입력
    start_date_value = datetime.strptime(params["start_date"], '%Y-%m-%d').date()
    start_date = st.date_input("투자시작일 *", value=start_date_value)
    if start_date.strftime('%Y-%m-%d') != params["start_date"]:
        params["start_date"] = start_date.strftime('%Y-%m-%d')
        # 고유 ID와 표시 이름 모두 전달
        save_params_robust(params, UNIQUE_ID_KEY, CURRENT_DISPLAY_NAME)

with col4:
    # 📝 투자 종료일 입력 (이 값은 Sheets에 저장되지 않음)
    end_date_value = datetime.strptime(params["end_date"], '%Y-%m-%d').date() if params.get("end_date") and params["end_date"] else datetime.now().date()
    end_date = st.date_input("투자종료일 *", value=end_date_value)
    # NOTE: end_date는 save_params_robust에서 저장하지 않도록 유지합니다.

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------
# 방어모드 파라미터 (생략된 함수들은 오류 방지를 위해 임시 정의)
# ---------------------------------------
st.subheader("💹 방어모드 설정")
dfns_hold_days = selected_style["dfns_hold_days"]
dfns_buy_threshold = selected_style["dfns_buy_threshold"] / 100
dfns_sell_threshold = selected_style["dfns_sell_threshold"] / 100
dfns_div_cnt = selected_style["dfns_div_cnt"]

st.markdown(f"**최대보유일수:** {dfns_hold_days}일")
st.markdown(f"**분할수:** {dfns_div_cnt}회")

col5, col6 = st.columns(2)
with col5:
    st.markdown(f"**매수조건이율:** {selected_style['dfns_buy_threshold']}%")

with col6:
    st.markdown(f"**매도조건이율:** {selected_style['dfns_sell_threshold']}%")

st.markdown("<br>", unsafe_allow_html=True)
# ---------------------------------------
# 공격모드 파라미터
# ---------------------------------------
st.subheader("💹 공격모드 설정")
atck_hold_days = selected_style["atck_hold_days"]
atck_buy_threshold = selected_style["atck_buy_threshold"] / 100
atck_sell_threshold = selected_style["atck_sell_threshold"] / 100
atck_div_cnt = selected_style["atck_div_cnt"]

st.markdown(f"**최대보유일수:** {atck_hold_days}일")
st.markdown(f"**분할수:** {atck_div_cnt}회")

col7, col8 = st.columns(2)
with col7:
    st.markdown(f"**매수조건이율:** {selected_style['atck_buy_threshold']}%")

with col8:
    st.markdown(f"**매도조건이율:** {selected_style['atck_sell_threshold']}%")

st.markdown("<br>", unsafe_allow_html=True)

# --- 전략 실행 버튼 이후의 로직 (간소화) ---

# 더미 함수 정의 (원본 코드의 오류 방지용)
#def get_mode_and_target_prices(*args): return pd.DataFrame()
#def extract_orders(*args): return [], []
#def remove_duplicates(*args): pass
#def print_table(*args): return pd.DataFrame()
#def highlight_order(*args): return pd.DataFrame()

if st.button("▶ 전략 실행"):
    if start_date > end_date:
        st.error("시작일은 종료일보다 이전이어야 합니다.")
        st.stop()
        
    status_placeholder = st.empty()
    status_placeholder.info("전략 실행 중입니다. (데이터 로드 및 계산에 시간이 걸릴 수 있습니다.)")

    prft_cmpnd_int_rt = selected_style["prft_cmpnd_int_rt"]
    loss_cmpnd_int_rt = selected_style["loss_cmpnd_int_rt"]

    # 캐싱된 함수 호출 시 모든 인자 전달
    df_result = get_mode_and_target_prices(
        start_date, end_date, target_ticker, first_amt, 0, 
        dfns_hold_days, dfns_buy_threshold, dfns_sell_threshold, dfns_div_cnt, 
        atck_hold_days, atck_buy_threshold, atck_sell_threshold, atck_div_cnt, 
        prft_cmpnd_int_rt, loss_cmpnd_int_rt
    )

    pd.set_option('future.no_silent_downcasting', True)

    printable_df = df_result.replace({None: np.nan})
    printable_df = printable_df.astype(str).replace({"None": "", "nan": ""})

    if printable_df.empty:
        status_placeholder.empty()
        st.warning("데이터가 없습니다. 입력 조건을 확인하세요.")
    else:
        status_placeholder.empty()
        st.success("전략 실행 완료!")
        
        # --- 요약 계산 로직 ---
        buy_data = df_result[["일자", "매수가", "매수량"]].copy()
        buy_data.columns = ["date", "price", "quantity"]
        sell_data = df_result[["실제매도일", "실제매도가", "실제매도량"]].copy()
        sell_data.columns = ["date", "price", "quantity"]
        sell_data = sell_data.dropna(subset=["quantity"])
        sell_data["quantity"] = -sell_data["quantity"]

        # 수정 코드: 비어있지 않은 데이터프레임만 병합
        dataframes_to_concat = []
        if not buy_data.empty:
            dataframes_to_concat.append(buy_data)
        if not sell_data.empty:
            dataframes_to_concat.append(sell_data)

        if dataframes_to_concat:
            df = pd.concat(dataframes_to_concat, ignore_index=True)
        else:
            # 둘 다 비어있을 경우, 컬럼 구조를 유지하며 빈 DF 생성
            df = pd.DataFrame(columns=buy_data.columns)

        df = df.dropna(subset=["date", "price", "quantity"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        avg_prc = 0.0
        if not df.empty:
            history = []
            unique_dates = df["date"].unique()

            for trade_date in unique_dates:
                sub = df[df["date"] == trade_date]
                p = sub["price"].iloc[0]
                q = sub["quantity"].sum()
                past_qty = df[df["date"] < trade_date]["quantity"].sum()

                if avg_prc == 0:
                    avg_prc = p
                elif q < 0: # 매도일 경우 평단 유지
                    pass
                else: # 매수일 경우 가중평균
                    if (past_qty + q) > 0:
                        avg_prc = (avg_prc * past_qty + p * q) / (past_qty + q)
                history.append((trade_date.date(), round(avg_prc, 4)))

        total_qty = int(df["quantity"].sum())
        # 매수/매도 금액이 모두 있는 행만 대상으로 손익 계산
        total_profit = df_result.dropna(subset=["실제매도금액", "매수금액"]).apply(
            lambda row: (row["실제매도금액"] - row["매수금액"]), axis=1
        ).sum()
        profit_ratio = (total_profit / first_amt * 100) if first_amt else 0
        
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

        # --- 매매 리스트 및 다운로드 ---
        styled_df = printable_df.style.format({
            "종가": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "변동률": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "매수예정": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            "LOC매수목표": lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) and str(x).strip() != "" else "",
            #"LOC매수목표": lambda x: "{:,.2f}".format(float(x)) if (pd.notnull(x) and not isinstance(x, (pd.Series, pd.DataFrame))) else "",
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

        # --- 당일 주문 리스트 (상계 처리) ---
        sell_orders, buy_orders = extract_orders(df_result)
        # print_orders(sell_orders, buy_orders) # 콘솔 출력 생략
        remove_duplicates(sell_orders, buy_orders)

        df_sell = print_table(sell_orders)
        df_buy = print_table(buy_orders)
        df_order_result = pd.concat([df_sell, df_buy], ignore_index=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 당일 주문 리스트")
        styled_df_orders = (df_order_result.reset_index(drop=True)
                            .style
                            .apply(highlight_order, axis=1).format({"주문가": "{:,.2f}"})
                        ) 
        st.dataframe(styled_df_orders, use_container_width=True)

        
