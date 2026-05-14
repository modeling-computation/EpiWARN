import pandas as pd

class Config:
    # 경로 설정
    DATA_PATH = './data/ILI_260126.xlsx'
    RESULT_PATH = './ILI_result/'
    MODEL_SAVE_PATH = './ILI_result/ILI_bootstrap_train_model_1000.joblib'

    # 데이터 관련 설정
    EPI_COL = 'ILI'
    SAMPLE_WINDOW = 12
    OUTBREAK_SEASON = 9  # 9월
    
    # 날짜 설정
    COVID_START = '2020-08-01'
    COVID_END = '2022-05-30'
    
    TRAIN_DATE = '2024-09-15'
    START_TEST_DATE = '2024-09-22'
    TEST_DATE = '2025-08-10'
    
    # 분석 기간 필터링 (데이터 로드 시 사용)
    START_ANALYSIS_DATE = '2017-01-01'

    # 알고리즘 설정
    BOOT_NUM = 1000      # 부트스트랩 횟수
    TEST_STEP = 1        # 테스트 데이터 증가 간격
    
    # KDCA / CUSUM 비교용 날짜 리스트
    START_EPIDEMIC = [
        pd.to_datetime('2015-01-22'), pd.to_datetime('2016-01-14'), 
        pd.to_datetime('2016-12-08'), pd.to_datetime('2017-12-03'), 
        pd.to_datetime('2018-11-18'), pd.to_datetime('2019-11-17'), 
        pd.to_datetime('2022-09-18'), pd.to_datetime('2023-09-17'), 
        pd.to_datetime('2024-12-22')
    ]
    
    CPD_DATE = [
        pd.to_datetime('2017-12-17'), pd.to_datetime('2018-12-02'), 
        pd.to_datetime('2019-12-08'), pd.to_datetime('2022-12-11'), 
        pd.to_datetime('2023-10-22'), pd.to_datetime('2024-12-22')
    ]