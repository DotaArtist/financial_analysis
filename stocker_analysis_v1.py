#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python version 3.9.10


import os
import talib
import math
import time
import json
import configparser
import numpy as np
import baostock as bs
import pandas as pd
from tqdm import tqdm
from datetime import timedelta, date
import mplfinance as mpf
import matplotlib.pyplot as plt
from functools import wraps


def calculate_time(place=2):
    """计算函数运行时间装饰器:param place: 显示秒的位数，默认为2位"""
    def decorator(func):
        """时间函数"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            """参数解析"""
            beg = time.time()
            f = func(*args, **kwargs)
            end = time.time()
            s = '{}()：{:.%sf} s' % place
            # print(s.format("耗时", end - beg))
            return f
        return wrapper
    return decorator


class BasicData(object):
    """基础数据"""
    def __init__(self, overwrite):
        self.data_path = "./data"
        self.past_duration = 180
        self.overwrite = overwrite
        self.today = date.today().strftime("%Y-%m-%d")
        self.yesterday = (date.today() + timedelta(-1)).strftime("%Y-%m-%d")
        self.past_180_day = (date.today() + timedelta(-self.past_duration)).strftime("%Y-%m-%d")
        self.recent_180days = []
        for i in range(self.past_duration):
            day = (date.today() + timedelta(-i)).strftime("%Y-%m-%d")
            self.recent_180days.append(day)

    def get_trade_day(self):
        """交易日信息"""
        if self.overwrite:
            lg = bs.login()
            print('### 登陆 error_msg:' + lg.error_msg)
            rs = bs.query_trade_dates(start_date=self.past_180_day, end_date=self.yesterday)
            print('### 交易日查询 error_msg:' + rs.error_msg)
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            _result = pd.DataFrame(data_list, columns=rs.fields)
            _result.to_csv(os.path.join(self.data_path, "trade_datas.csv"), encoding="gbk", index=False)
            bs.logout()
            return _result
        else:
            _result = pd.read_csv(os.path.join(self.data_path, "trade_datas.csv"), encoding="gbk")
            return _result

    def get_industry_data(self):
        """申万行业数据"""
        if self.overwrite:
            lg = bs.login()
            print('### 登陆 error_msg:' + lg.error_msg)
            rs = bs.query_stock_industry(date=self.yesterday)
            print('### 行业分类查询 error_msg:' + rs.error_msg)
            industry_list = []
            while (rs.error_code == '0') & rs.next():
                industry_list.append(rs.get_row_data())
            _result = pd.DataFrame(industry_list, columns=rs.fields)
            _result.to_csv(os.path.join(self.data_path, "stock_industry.csv"), encoding="gbk", index=False)
            bs.logout()
            return _result
        else:
            _result = pd.read_csv(os.path.join(self.data_path, "stock_industry.csv"), encoding="gbk")
            return _result

    def get_all_stock(self):
        """获取股票清单"""
        if self.overwrite:
            lg = bs.login()
            print('### 登陆 error_msg:' + lg.error_msg)
            rs = bs.query_all_stock(day=self.yesterday)
            print('### 股票清单 error_msg:' + rs.error_msg)
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            _result = pd.DataFrame(data_list, columns=rs.fields)
            _result.to_csv(os.path.join(self.data_path, "all_stock.csv"), encoding="gbk", index=False)
            bs.logout()
        else:
            _result = pd.read_csv(os.path.join(self.data_path, "all_stock.csv"), encoding="gbk")
            return _result

    @calculate_time(place=2)
    def download_daily_data_all(self, trans_date):
        """获取日数据"""
        if not os.path.exists(os.path.join(self.data_path, "daily/{}.csv".format(trans_date))):
            bs.login()
            stock_rs = bs.query_all_stock(trans_date)
            stock_df = stock_rs.get_data()
            frames = []
            try:
                with tqdm(stock_df["code"], position=0, leave=True) as t:
                    for code in t:
                        k_rs = bs.query_history_k_data_plus(code, "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                                                            trans_date, trans_date, frequency="d", adjustflag="2")
                        frames.append(k_rs.get_data())
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            data_df = pd.concat(frames)
            bs.logout()
            if data_df.shape[0] > 0:
                data_df.to_csv(os.path.join(self.data_path, "daily/{}.csv".format(trans_date)), encoding="gbk", index=False)
        else:
            print("### 数据已存在 {} !".format(trans_date))

    def download_recent_daily_data(self):
        """近期k线数据下载"""
        trade_day = self.get_trade_day()
        trade_day_list = trade_day[trade_day["is_trading_day"] == '1']["calendar_date"].tolist()
        for i in tqdm(trade_day_list[::-1]):
            print("### 数据正在下载 {} ...".format(i))
            self.download_daily_data_all(trans_date=i)

    def read_recent_daily_data(self):
        """近期数据读取"""
        datas = []
        for trans_date in self.recent_180days:
            if os.path.exists(os.path.join(self.data_path, "daily/{}.csv".format(trans_date))):
                tmp = pd.read_csv(os.path.join(self.data_path, "daily/{}.csv".format(trans_date)))
                datas.append(tmp)
        _result = pd.concat(datas)
        return _result


class BasicPlots(object):
    """基础绘图"""
    @staticmethod
    def golden_crossing(df):
        """DIFF线低于DEA"""
        maker = [np.nan] * df.shape[0]
        for _, __ in enumerate(zip(df["DIFF"], df["DEA"])):
            i, j = __
            if i >= j and df["DIFF"].iloc[_ - 1] < df["DEA"].iloc[_ - 1]:
                maker[_] = df["low"].iloc[_] * 0.98
        return maker

    @staticmethod
    def dark_crossing(df):
        """DIF线高于DEA
        DIF = EMA12 - EMA26；
        EMA = 昨日EMA + (收盘价 - 昨日EMA) * 2 / (N + 1)，N代表的是所选定的周期数；
        DEA（DEA为DIF的9天指数平滑移动平均线）= 前一日DEA + (当前DIF - 前一日DEA) × 2 / (9 + 1)
        """
        maker = [np.nan] * df.shape[0]
        for _, __ in enumerate(zip(df["DIFF"], df["DEA"])):
            i, j = __
            if i <= j and df["DIFF"].iloc[_ - 1] > df["DEA"].iloc[_ - 1]:
                maker[_] = df["high"].iloc[_] * 1.02
        return maker

    def plot_kline(self, _result, code, duration=60, name="", save_img=False):
        """日线绘制"""
        df = _result[_result["code"] == code]
        df = df[::-1]
        df.set_index(["date"], inplace=True)
        df.index = pd.to_datetime(df.index)

        df['DIFF'], df['DEA'], df['MACD'] = talib.MACD(df['close'].values)
        df['MACD'] *= 2

        high_signal = self.golden_crossing(df)
        low_signal = self.dark_crossing(df)
        signal = "neu"  # 数据不全
        for i, ii in enumerate(zip(high_signal[::-1], low_signal[::-1])):
            j, k = ii
            if not math.isnan(j):
                signal = "pos_{}".format(i)
                break
            if not math.isnan(k):
                signal = "neg_{}".format(i)
                break
        if save_img:
            rcp_dict = {'font.family': ['AR PL UKai CN', 'sans-serif']}
            style = mpf.make_mpf_style(marketcolors=mpf.make_marketcolors(up="#C70039", down="#28B463", inherit=True),
                                       gridcolor="gray", gridstyle="--", gridaxis="both", rc=rcp_dict)
            start = max(df.shape[0] - duration, 0)
            added_plots = {
                "MACD": mpf.make_addplot(df['MACD'].iloc[start:], color="black", linewidths=0.1, linestyle='--'),
                "DIFF": mpf.make_addplot(df['DIFF'].iloc[start:], color="#F57F17", linewidths=0.1, linestyle='solid'),
                "DEA": mpf.make_addplot(df['DEA'].iloc[start:], color="#FDD835", linewidths=0.1, linestyle='solid'),
                "seller": mpf.make_addplot(low_signal[start:], type='scatter', color="#004D40", linewidths=1, markersize=50, marker='v'),
                "buyer": mpf.make_addplot(high_signal[start:], type='scatter', color="#B71C1C", linewidths=1, markersize=50, marker='^', ylabel="high"),
                "TURN": mpf.make_addplot((df['turn'].iloc[start:]), panel=1, color='black', linestyle='dotted', ylabel="turn")
            }
            # fig, axes = mpf.plot(df.iloc[start:], type="candle", style=style, volume=True, block=False, returnfig=True)
            fig, axes = mpf.plot(df.iloc[start:], type="candle", style=style, addplot=list(added_plots.values()), volume=True, block=False, returnfig=True)
            axes[0].legend([None] * (len(added_plots) + 2))
            handles = axes[0].get_legend().legendHandles
            axes[0].legend(handles=handles, labels=list(added_plots.keys()), loc='lower left')
            axes[0].set_title('{}'.format(code), fontsize=15, fontfamily='fantasy', loc='center')
            axes[0].set_ylabel("Price [KRW]")
            axes[2].set_ylabel("Volume [ea]")
            fig.savefig('./data/png/{}.png'.format(code))
            plt.close(fig)
        return df, signal, df["pctChg"].iloc[-1], df["turn"].iloc[-1], df["amount"].iloc[-1] / 1000 / 1000


class StockerAnalysisV1(object):
    """v1: k线分析"""
    def __init__(self, _data_module, _plot_module):
        self.today = date.today().strftime("%Y-%m-%d")

        self.config = configparser.ConfigParser()
        self.config.read("关注.ini")
        self.data_module = _data_module
        self.plot_module = _plot_module

    def static_analysis(self):
        """统计分析"""
        all_stock = self.data_module.get_all_stock(overwrite=False)
        _result = self.data_module.read_recent_daily_data()
        industry_data = self.data_module.get_industry_data()

        data = []
        print("### 统计分析...")
        for i in tqdm(all_stock.loc[all_stock["tradeStatus"] == 1]["code"]):
            """sz.3--- 创业板 sh.68-- 科创板 sh.0--- 指数 bj.---- 北交所"""
            if "sh.60" in i or "sz.00" in i or i in json.loads(self.config["focus"]["star"]):
                try:
                    df, signal, pctChg, turn, amount = self.plot_module.plot_kline(_result=_result, code=i, name="", save_img=False)
                    data.append([i, signal, pctChg, turn, amount])
                except ValueError:
                    print(i, "ValueError")
                except Exception as e:
                    print(i, e)
        df = pd.DataFrame(data, columns=['code', '趋势', '收盘涨跌', '换手率', '交易金额(百万)'])
        with pd.ExcelWriter("./data/分析/{}.xlsx".format(self.today), engine='xlsxwriter') as writer:
            df_ana = df.merge(industry_data, left_on='code', right_on='code')[['code', '趋势', '收盘涨跌', '换手率', '交易金额(百万)', 'code_name', 'industry']]
            df_ana["市值(亿)"] = df_ana.apply(lambda x: x["交易金额(百万)"] / x["换手率"], axis=1)
            df_ana["涨跌"] = df_ana[["趋势"]].map(lambda x: "涨" if "pos" in str(x) else "跌")
            df_ana["强信号"] = df_ana[["趋势"]].map(lambda x: "大涨" if "pos_0" in str(x) else "无")
            df_ana["弱信号"] = df_ana[["趋势"]].map(lambda x: "大跌" if "neg_0" in str(x) else "无")
            df_ana.to_excel(writer, sheet_name="个股趋势")

            df_static = df_ana.pivot_table(index=["industry"], values=["涨跌", "强信号", "弱信号"], aggfunc=lambda x: x.value_counts().to_dict())
            df_static["涨率"] = df_static[["涨跌"]].map(lambda x: '{:.2%}'.format(x.get("涨", 0) / (x.get("涨", 0) + x.get("跌", 0))))
            df_static["大涨率"] = df_static[["强信号"]].map(lambda x: '{:.2%}'.format(x.get("大涨", 0) / (x.get("大涨", 0) + x.get("无", 0))))
            df_static["大跌率"] = df_static[["弱信号"]].map(lambda x: '{:.2%}'.format(x.get("大跌", 0) / (x.get("大跌", 0) + x.get("无", 0))))
            df_static["涨数"] = df_static[["涨跌"]].map(lambda x: x.get("涨", 0))
            df_static["跌数"] = df_static[["涨跌"]].map(lambda x: x.get("跌", 0))
            df_static["大涨数"] = df_static[["强信号"]].map(lambda x: x.get("大涨", 0))
            df_static["大跌数"] = df_static[["弱信号"]].map(lambda x: x.get("大跌", 0))
            df_static["股票数"] = df_static[["涨跌"]].map(lambda x: (x.get("涨", 0) + x.get("跌", 0)))
            df_static = df_static[["涨率", "大涨率", "涨数", "大涨数", "跌数", "大跌数", "股票数"]].sort_values("涨率", ascending=False)
            df_static.to_excel(writer, sheet_name="行业趋势")

    def label_analysis(self):
        """异常洞察"""
        windows_size = 90
        _result = self.data_module.read_recent_daily_data()
        industry_data = self.data_module.get_industry_data()
        new_result = pd.merge(left=_result, right=industry_data, on="code", how="inner").dropna()
        day_filter = list(new_result.date.unique()[:windows_size])  # 日期过滤
        _tmp_code = new_result[new_result.date.isin(day_filter)].code.value_counts()
        code_filter = list(_tmp_code[_tmp_code == windows_size].index)  # 代码过滤
        new_result_filtered = new_result[new_result.code.isin(code_filter) & new_result.date.isin(day_filter)]
        grouped_sorted = new_result_filtered.sort_values(by=['code', 'date'], ascending=True).groupby('code')['close'].apply(list).reset_index(name='close_list')
        pass


if __name__ == '__main__':
    data_module = BasicData(overwrite=False)
    data_module.download_recent_daily_data()
    result = data_module.read_recent_daily_data()

    plot_module = BasicPlots()
    # plot_module.plot_kline(result, "sh.600970", duration=60, name="", save_img=True)

    analysis_module = StockerAnalysisV1(_data_module=data_module, _plot_module=plot_module)
    # analysis_module.static_analysis()
    analysis_module.label_analysis()
    pass
