import backtrader as bt
from config import config
import logging
from datetime import date
from typing import Tuple, Any
from drl_agent import DRLAgent

class BuyAndHold(bt.Strategy):

    def __init__(self):
        self.triggs_count: int = 0

    def next(self):
        try:
            self.data0.open[2]
        except IndexError:
            if self.position:
                self.sell()
            return
        if not self.position:
            self.buy()

class TrailingStop(bt.Strategy):
    params = (('stop_pct', .5),)

    def __init__(self):
        self.triggs_count: int = 0

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.exectype == order.Stop:
                self.triggs_count += 1
                
    def next(self):
        try:
            self.data0.open[2]
        except IndexError:
            if self.position:
                self.sell_order = self.sell()
            return
        stop_price = self.data0.open[1] * (100 - self.params.stop_pct)/100
        if not self.position:
            self.buy_order = self.buy(transmit=False)
            self.stop_order = self.sell(exectype=bt.Order.Stop, 
                                        price=stop_price, 
                                        parent=self.buy_order)
        else:
            self.cancel(self.stop_order)
            self.stop_order = self.sell(exectype=bt.Order.Stop, 
                                        price=stop_price)
            

class FixedStop(bt.Strategy):
    params = (('stop_pct', .5),)
    
    def __init__(self):
        self.triggs_count: int = 0

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.exectype == order.Stop:
                self.triggs_count += 1
                
    def next(self):
        try:
            self.data0.open[2]
        except IndexError:
            if self.position:
                self.sell_order = self.sell()
            return
        if not self.position:
            stop_price = self.data0.open[1] * (100 - self.params.stop_pct)/100
            self.buy_order = self.buy(transmit=False)
            self.stop_order = self.sell(exectype=bt.Order.Stop, 
                                        price=stop_price, 
                                        parent=self.buy_order)


class DRLStrategy(bt.Strategy):
    params: Tuple[Tuple[str, Any], ...] = (
        ('agent', None),
        ('train', True),
    )

    def __init__(self):
        # self.broker.set_coc(True)
        self.agent: DRLAgent = self.params.agent
        self.train: bool = self.params.train
        self.actions: list[int] = []
        self.step: int = 0
        self.diffs: list[float] = []
        self.strategy_returns: list[float] = []
        self.market_returns: list[float] = []
        self.stop_order: bt.Order|None = None
        self.sell_order: bt.Order|None = None
        self.buy_order: bt.Order|None = None
        self.buy_count: int = 0
        self.sell_count: int = 0
        self.triggs: list[float] = []
        self.first_open: float|None = None
        self.last_open: float|None = None
        self.triggs_count: int = 0

        # Technical indicators
        self.sma2 = bt.indicators.MovingAverageSimple(period=2)
        self.sma5 = bt.indicators.MovingAverageSimple(period=5)
        self.sma10 = bt.indicators.MovingAverageSimple(period=10)
        self.sma20 = bt.indicators.MovingAverageSimple(period=20)
        self.sma30 = bt.indicators.MovingAverageSimple(period=30)
        self.std2 = bt.indicators.StdDev(period=2)
        self.std5 = bt.indicators.StdDev(period=5)
        self.std10 = bt.indicators.StdDev(period=10)
        self.std20 = bt.indicators.StdDev(period=20)
        self.std30 = bt.indicators.StdDev(period=30)

        # OHLCV data
        self.open = self.data0.open
        self.high = self.data0.high
        self.low =  self.data0.low
        self.close = self.data0.close
        self.volume = self.data0.volume
        self.dt = self.data0.datetime


    def log(self, txt: str, dt: None|date = None):
        dt = dt or self.datas[0].datetime.datetime(0)
        logging.info('%s: %s' % (dt.isoformat(), txt))

    def _get_dql_observation(self, idx: int = 0) -> list[float]:
        vol_prev = 1 if self.volume[idx-1] == 0 else self.volume[idx-1]
        return [
            self.sma2[idx]/self.close[idx],
            self.sma5[idx]/self.close[idx],
            self.sma10[idx]/self.close[idx],
            self.sma20[idx]/self.close[idx],
            self.sma30[idx]/self.close[idx],
            self.std2[idx]/self.close[idx],
            self.std5[idx]/self.close[idx],
            self.std10[idx]/self.close[idx],
            self.std20[idx]/self.close[idx],
            self.std30[idx]/self.close[idx],
            self.open[idx]/self.close[idx],
            self.high[idx]/self.close[idx],
            self.low[idx]/self.close[idx],
            self.close[idx]/self.close[idx-1],
            (self.volume[idx] - vol_prev)/vol_prev,
        ]

    def memorize_transition(self) -> None:
        market_return = (self.open[1] - self.open[0])/self.open[0] * 100
        trigg = None
        if self.actions[-1] == 0: # The asset was sold or not bought on today's open
            strategy_return = .0
        elif self.stop_order.status == bt.Order.Completed: # Stop-loss was triggered
            trigg = True
            strategy_return = (self.stop_order.executed.price - self.open[0])/self.open[0] * 100
        else : # Stop-loss order was created but was not triggered during the previous day
            strategy_return = market_return # Strategy return is same as market's
            trigg = False
        diff = strategy_return - market_return
        self.triggs.append(1.0 if trigg == True else .0)
        self.market_returns.append(market_return)
        self.strategy_returns.append(strategy_return)
        self.diffs.append(diff)
        if self.train:
            self.agent.memorize_transition(
                self._get_dql_observation(-1),
                self.actions[-1],
                diff,
                self._get_dql_observation(0),
                not_done=True
            )
            self.agent.experience_replay()
        self.log(f'Experienced transition: Stop trigg: {trigg}, Close: {self.close[-1]:.2f}, Close`: {self.close[0]:.2f}, Action: {self.actions[-1]}, Reward: {diff:.2f}')

    def notify_order(self, order: bt.Order) -> None:
        if order.status == order.Completed:
            if order.issell():
                self.sell_count += 1
            else:
                self.buy_count += 1
        order_type = 'BUY' if order.isbuy() else ('STOP' if order.exectype == order.Stop else 'SELL')
        if order.status == order.Expired:
            self.log(f'{order_type} ORDER EXPIRED')
        elif order.status == order.Completed:
            self.log(f'{order_type} EXECUTED AT {order.executed.price:.2f} X {order.size}')
            if order.exectype == order.Stop:
                self.triggs_count += 1
        elif order.status == order.Created:
            self.log(f'{order_type} CREATED WITH {order.price:.2f} X {order.size}')
        elif order.status == order.Submitted:
            price_f = f'{order.price:.2f}' if order.price is not None else '-'
            self.log(f'{order_type} SUBMITTED WITH {price_f} X {order.size}')
        elif order.status == order.Canceled:
            self.log(f'{order_type} CANCELED')
        elif order.status == order.Accepted:
            self.log(f'{order_type} ACCEPTED')
        elif order.status == order.Rejected:
            self.log(f'{order_type} REJECTED')
        elif order.status == order.Margin:
            self.log(f'{order_type} MARGIN')
            if not order.isbuy():
                raise Exception('Margin not allowed')
            print('margin')

    def stop(self):
        self.last_open = self.open[0]

    def next(self):
        if self.step == 0:
            self.first_open = self.open[1]
        if self.train and self.step > 0 and self.step <= config['trading_days']:
            self.memorize_transition()
        self.step += 1
        self.log(
            f'Open:{self.open[0]:.2f}, ' +
            f'High:{self.high[0]:.2f}, ' +
            f'Low:{self.low[0]:.2f}, ' +
            f'Close:{self.close[0]:.2f}, ' +
            f'Volume:{self.volume[0]:.2f}')
        self.cancel(self.sell_order)
        self.cancel(self.buy_order)
        self.cancel(self.stop_order)
        if (self.train and self.step > config['trading_days']): 
            if self.position:
                self.sell_order = self.sell()
            return
        if not self.train:
            try:
                self.open[2]
            except IndexError:
                if self.position:
                    self.sell_order = self.sell()
                return
        obs = self._get_dql_observation()
        action = self.agent.epsilon_greedy_policy(obs) if self.train else self.agent.predict(obs)
        self.actions.append(action)
        self.log(action)
        if action != 0:
            stop_price = self.open[1] * (100 - config['stop_pct'][action-1])/100
            if not self.position:
                self.buy_order = self.buy(transmit=False)
                self.stop_order = self.sell(exectype=bt.Order.Stop, 
                                            price=stop_price, 
                                            parent=self.buy_order)
            else:
                self.stop_order = self.sell(exectype=bt.Order.Stop,
                                            price=stop_price)
        elif action == 0 and self.position:
            self.sell_order = self.sell()
