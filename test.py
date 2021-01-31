import tushare as ts

pro = ts.pro_api()

# 交易日历信息
df = pro.query('trade_cal', exchange='', start_date='20180901', end_date='20181001',
               fields='exchange,cal_date,is_open,pretrade_date', is_open='0')

# 每日行情
df = ts.get_today_all()
