from jqdatasdk import *
import pandas as pd
from datetime import datetime


auth("13818134484", "12345678")

# 查询剩余可调用次数
get_query_count()

stocks = get_all_securities(['stock'])
stocks['code'] = stocks.index.values

stocks['market'] = stocks[['code']].applymap(lambda x:x.split('.')[1])
stocks['stock_code'] = stocks[['code']].applymap(lambda x:x.split('.')[0])

today = datetime.now()
today_str = datetime.now().strftime('%Y-%m-%d')
stocks['是否退市'] = stocks[['end_date']].applymap(lambda x:1 if x < today else 0)

# 板块类别数量：zjw,90;sw_l1,28;sw_l2,104;sw_l3,227

stock_list = stocks[stocks['是否退市'] == 0]['code'].tolist()
d = get_industry(security=stock_list, date=today_str)

# 板块信息
industry_block_type = list(d[list(d.keys())[0]].keys())

def get_industry_block(x, y):
    try:
        return d[x][y]['industry_name']
    except KeyError:
        return 'null'

for i in industry_block_type:
    stocks[i] = stocks[['code']].applymap(lambda x:get_industry_block(x,i))


# 行情数据
m = get_price(stock_list, start_date='2021-01-01', end_date=today_str,
 frequency='daily', fq='pre')

# 集合竞价
call_data = get_call_auction(stock_list, start_date='2021-01-01', end_date=today_str)