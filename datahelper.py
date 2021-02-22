#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据采集模块"""

__author__ = 'yp'

import h5py
import pytz
import time
from jqdatasdk import *
from datetime import datetime, timedelta
from mylogparse import LogParse
from apscheduler.schedulers.background import BackgroundScheduler

auth("13818134484", "12345678")
print(get_query_count())

a = LogParse()
a.set_profile(path=".", filename="log")


# 数据存储
def save_jqdata(stock_code, target_day_str):
    dt = h5py.string_dtype('utf-8', 30)

    with h5py.File('D:/data_file/jqdata.hdf5', 'a') as files:

        if target_day_str in files.keys():
            grp = files[target_day_str]
        else:
            grp = files.create_group(target_day_str)

        if stock_code in grp.keys():
            subgrp = grp[stock_code]
        else:
            subgrp = grp.create_group(stock_code)

        # 每日行情数据
        if "price_daily" not in subgrp.keys():
            subgrp_daily = get_price(stock_code, start_date=target_day_str, end_date=target_day_str, frequency='daily',
                                     fq='post')
            if subgrp_daily.shape[0] > 0:
                subgrp.create_group("price_daily")

                subgrp["price_daily"].create_dataset("columns",
                                                     data=list(subgrp_daily.columns), dtype=dt)
                subgrp["price_daily"].create_dataset("values", data=subgrp_daily.values.tolist())

        else:
            a.info('写入失败.日期:{}.代码:{}.每日行情数据已存在.'.format(target_day_str, stock_code))

        # 每日分钟数据
        if "price_minute" not in subgrp.keys():
            subgrp_minute = get_price(stock_code, start_date=target_day_str,
                                      end_date=(datetime.strptime(target_day_str, '%Y-%m-%d') + timedelta(1)).strftime('%Y-%m-%d'),
                                      frequency='minute', fq='post')
            if subgrp_minute.shape[0] == 240:
                subgrp.create_group("price_minute")

                subgrp["price_minute"].create_dataset("columns",
                                                      data=list(subgrp_minute.columns), dtype=dt)
                subgrp["price_minute"].create_dataset("values", data=subgrp_minute.values.tolist())

        else:
            a.info('写入失败.日期:{}.代码:{}.每日分钟数据已存在.'.format(target_day_str, stock_code))

        # 每日集合竞价
        if "price_call" not in subgrp.keys():
            subgrp_call = get_call_auction(stock_code, start_date=target_day_str, end_date=target_day_str)
            del subgrp_call['time']
            del subgrp_call['code']
            if subgrp_call.shape[0] > 0:  # 集合竞价为空
                subgrp.create_group("price_call")

                subgrp["price_call"].create_dataset("columns",
                                                    data=list(subgrp_call.columns), dtype=dt)
                subgrp["price_call"].create_dataset("values", data=subgrp_call.values.tolist())

        else:
            a.info('写入失败.日期:{}.代码:{}.每日集合竞价数据已存在.'.format(target_day_str, stock_code))

        # 每日融资融券
        if "price_mtss" not in subgrp.keys():
            subgrp_mtss = get_mtss(stock_code, start_date=target_day_str, end_date=target_day_str)
            del subgrp_mtss['date']
            del subgrp_mtss['sec_code']
            if subgrp_mtss.shape[0] > 0:  # 无融资融券
                subgrp.create_group("price_mtss")

                subgrp["price_mtss"].create_dataset("columns",
                                                    data=list(subgrp_mtss.columns), dtype=dt)
                subgrp["price_mtss"].create_dataset("values", data=subgrp_mtss.values.tolist())

        else:
            a.info('写入失败.日期:{}.代码:{}.每日融资融券数据已存在.'.format(target_day_str, stock_code))
    a.info('执行完成.日期:{}.代码:{}.'.format(target_day_str, stock_code))


# 每日数据存储
def save_jqdata_daily(stock_list, target_day_str):
    for _code in stock_list:
        save_jqdata(stock_code=_code, target_day_str=target_day_str)


def execute_update_jqdata():
    # 获取交易日数据
    trade_days = get_all_trade_days()
    trade_days_str = [i.strftime('%Y-%m-%d') for i in list(trade_days)]

    today = datetime.now()
    today_str = datetime.now().strftime('%Y-%m-%d')

    # 当日前一个交易日作为目标
    for _, i in enumerate(trade_days_str):
        if today_str <= i:
            target_day_str = trade_days_str[_ - 1]
            break

            # 目标日与前一个交易日间隔天数
    target_day_pre_delta = (
            trade_days[trade_days_str.index(target_day_str) - 1] - trade_days[trade_days_str.index(target_day_str) - 2]).days

    # 获取未退市清单
    stocks = get_all_securities(['stock'])
    stocks['code'] = stocks.index.values

    stocks['market'] = stocks[['code']].applymap(lambda x: x.split('.')[1])
    stocks['stock_code'] = stocks[['code']].applymap(lambda x: x.split('.')[0])

    stocks['是否退市'] = stocks[['end_date']].applymap(lambda x: 1 if x < today else 0)

    stock_list = stocks[stocks['是否退市'] == 0]['code'].tolist()

    # 往前回溯28天数据
    for i in trade_days_str[trade_days_str.index(target_day_str)-20:trade_days_str.index(target_day_str)-1]:
        save_jqdata_daily(stock_list, i)
        # save_jqdata_daily(stock_list, '2021-01-04')
        # assert 1 == 2


if __name__ == '__main__':
    execute_update_jqdata()
    # reinitialze_scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Shanghai'))
    # reinitialze_scheduler.add_job(execute_update_jqdata, 'cron',
    #                               hour='23', minute='0', second='0')
    # reinitialze_scheduler.start()
    #
    # while True:
    #     time.sleep(60)
