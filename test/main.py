# -*- coding: utf-8 -*-

import os
import sys
import time
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import ticker


def read_data():
    data = pd.read_csv(f'601021.SH.csv', index_col=0, parse_dates=[0])
    return data

def main():
    data = read_data()
    print(data.head())

    # 计算过去120分钟max(high)序列，同样计算min(low)系列
    h = data['high']
    l = data['low']
    c = data['close']
    o = data['open']
    idx = data.index

    n = 120
    h_max = h.rolling(n).max().shift()
    l_min = l.rolling(n).min().shift()
    h = h.values
    l = l.values

    # 根据high、low价格和计算出的max、min序列，计算目标仓位序列pos
    pos = np.zeros_like(h)
    last_h = np.nan
    last_l = np.nan
    holding = 0   # 当前持仓方向
    h_max = h_max.values
    l_min = l_min.values
    for i, j in enumerate(c):  # 此处一般会写个循环来处理复杂逻辑；如果逻辑简单，也可以直接向量计算，省略循环
        if holding == 0:
            if h[i] >= h_max[i] and last_h < h_max[i - 1]:
                holding = 1
        elif holding == 1:
            if l[i] <= l_min[i] and last_l > l_min[i - 1]:
                holding = 0
        pos[i] = holding    # 更新pos序列
        last_l = l[i]
        last_h = h[i]

    chg = c.shift(-1) - c
    pnl = chg * pos
    total_return = np.cumsum(pnl)

    # 计算其他参数，交易次数，平均单笔收益，夏普值
    pos = pd.Series(pos, index=idx)
    open_point = ((pos != pos.shift()) & pos == 1).astype(int)
    trading_times = open_point.sum()
    # 使用分钟数据，夏普值要 × sqrt(一年的分钟数) = sqrt(一年245个交易日 * 每日交易时间240分钟)
    sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(245 * 240)
    each_return = np.sum(pnl) / trading_times

    print('----------------------------')
    print('trading_times: ', trading_times)
    print('each_return: ', each_return)
    print('sharpe: ', sharpe)
    print('----------------------------')

    # 收益曲线可视化
    def format_date(x, pos=None):   # 画时间不连续的序列
        thisind = np.clip(int(x), 0, len(idx) - 1)
        return idx[thisind].strftime('%Y-%m-%d %H:%M')

    plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(c.values, label='close')   # 这里的数据都是numpy数据，不带index
    plt.plot(h, label='high')
    plt.plot(l, label='low')
    plt.plot(h_max, label='max')
    plt.plot(l_min, label='min')
    plt.legend(loc=1)
    plt.xticks(rotation=20)
    plt.subplot(212, sharex=ax1)
    plt.plot(total_return.values, label='rate')
    plt.legend(loc=1)
    plt.xticks(rotation=20)

    # 替换index，使时间轴连续
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    plt.show()


if __name__ == '__main__':
    main()

