from datetime import datetime, timedelta
import pandas as pd
from config import config
import backtrader as bt
import numpy as np

def progress_info(step: int, max_steps: int, start: datetime,
                  diff_mean: float, actions_mean: float, 
                  losses_mean: float, epsilon: float,
                  episode: int, wins_pct: float,
                  buys_count: int, sell_count: int) -> None:
    """ Print progress info at the end of a step """
    percent_complete = step / max_steps * 100
    elapsed_time = (datetime.now() - start).total_seconds()
    estimated_total_time = (elapsed_time / step) * max_steps
    estimated_time_remaining = estimated_total_time - elapsed_time
    completion_datetime = datetime.now() + timedelta(seconds=estimated_time_remaining)
    completion_str = completion_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print("{:<13} | {:<15} | {:<24} | {:<10} | {:<15} | {:<20} | {:<15} | {:<15} | {:<15} | {:<30} | {:<30}".format(
    f"Episode: {episode}",
    f"Diff (MA100): {diff_mean:.2f}%",
    f"Losses (MA100): {losses_mean:.6f}",
    f"Epsilon: {epsilon:.2f}",
    f"Market beats: {wins_pct:.2f}%",
    f"Actions (MA100): {actions_mean:.2f}",
    f"Buys: {buys_count}",
    f"Sells: {sell_count}",
    f"Progress: {percent_complete:.2f}%",
    f"Est. time left: {estimated_time_remaining:.2f} sec",
    f"ETA: {completion_str}"
))
    
def create_ev_bt_feed(ticker_symbol: str, filename: str) -> bt.feeds.GenericCSVData:
    filename = filename.format(ticker_symbol)
    df = pd.read_csv(filename)
    df_len = len(df.index)
    req_len = 252 * config['eval_years'] + 31
    max_offset = df_len - req_len
    offset = max_offset
    return bt.feeds.GenericCSVData(
        dataname=filename,
        fromdate=pd.to_datetime(df.at[offset, "Date"]),
        todate=pd.to_datetime(df.at[offset+req_len-1, "Date"]),
        reverse=False,
        separator=',',
        timestamp=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
        timeframe=bt.TimeFrame.Days,
        dtformat='%Y-%m-%d %H:%M:%S%z',
    ),  df.at[offset+29, 'Close']

def create_tr_bt_feed(ticker_symbol: str, filename: str) -> bt.feeds.GenericCSVData:
    filename = filename.format(ticker_symbol)
    df = pd.read_csv(filename)
    df_len = len(df.index)
    req_len = config['trading_days'] + 31
    max_offset = df_len - req_len - config['eval_years'] * 252
    offset = np.random.randint(0, max_offset)
    return bt.feeds.GenericCSVData(
        dataname=filename,
        fromdate=pd.to_datetime(df.at[offset, "Date"]),
        todate=pd.to_datetime(df.at[offset+req_len, "Date"]),
        reverse=False,
        separator=',',
        timestamp=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
        timeframe=bt.TimeFrame.Days,
        dtformat='%Y-%m-%d %H:%M:%S%z',
    ), df.at[offset+29, 'Close']
