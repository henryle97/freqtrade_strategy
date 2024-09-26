import datetime

import pytz
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
import numpy as np
from freqtrade.strategy import IStrategy, informative


def calculate_angle(series, periods):
    radians = np.arctan((series.diff(periods) / periods).values)
    degrees = np.degrees(radians)
    return degrees


class SonicRStrategy(IStrategy):
    """
    SonicR strategy
    - Main timeframe: 4h
    - Context:
        + EMA34 > EMA89
        + 50 < RSI14 < 60
    - Signal:
        + Bullish candlestick pattern
        + Lowest price near ema34, close price > ema34
    """

    INTERFACE_VERSION = 3

    can_short: bool = False

    timeframe = "4h"
    ema: int = 34
    rsi: int = 14
    startup_candle_count: int = 200

    stoploss = -0.05

    # https://ta-lib.github.io/ta-lib-python/func_groups/pattern_recognition.html
    bullish_candlestick_patterns = {
        # "2 CROWS": ta.CDL2CROWS,
        # "3 BLACK CROWS": ta.CDL3BLACKCROWS,
        # "3 INSIDE": ta.CDL3INSIDE,
        # "3 LINE STRIKE": ta.CDL3LINESTRIKE,
        # "3 OUTSIDE": ta.CDL3OUTSIDE,
        # "3 STARS IN SOUTH": ta.CDL3STARSINSOUTH,
        # "3 WHITE SOLDIERS": ta.CDL3WHITESOLDIERS,
        # "ABANDONED BABY": ta.CDLABANDONEDBABY,
        # "ADVANCE BLOCK": ta.CDLADVANCEBLOCK,
        # "BELT HOLD": ta.CDLBELTHOLD,
        # "BREAKAWAY": ta.CDLBREAKAWAY,
        # "CLOSING MARUBOZU": ta.CDLCLOSINGMARUBOZU,
        # "CONCEAL BABY SWALL": ta.CDLCONCEALBABYSWALL,
        # "COUNTERATTACK": ta.CDLCOUNTERATTACK,
        # "DARK CLOUD COVER": ta.CDLDARKCLOUDCOVER,
        "DOJI": ta.CDLDOJI,
        # "DOJI STAR": ta.CDLDOJISTAR,
        # "DRAGONFLY DOJI": ta.CDLDRAGONFLYDOJI,
        "ENGULFING": ta.CDLENGULFING,
        # "EVENING DOJI STAR": ta.CDLEVENINGDOJISTAR,
        "EVENING STAR": ta.CDLEVENINGSTAR,
        # "GAP SIDE SIDE WHITE": ta.CDLGAPSIDESIDEWHITE,
        # "GRAVESTONE DOJI": ta.CDLGRAVESTONEDOJI,
        "HAMMER": ta.CDLHAMMER,
        "HANGING MAN": ta.CDLHANGINGMAN,
        # "HARAMI": ta.CDLHARAMI,
        # "HARAMI CROSS": ta.CDLHARAMICROSS,
        # "HIGH WAVE": ta.CDLHIGHWAVE,
        # "HIKKAKE": ta.CDLHIKKAKE,
        # "HIKKAKE MOD": ta.CDLHIKKAKEMOD,
        # "HOMING PIGEON": ta.CDLHOMINGPIGEON,
        # "IDENTICAL 3 CROWS": ta.CDLIDENTICAL3CROWS,
        # "IN NECK": ta.CDLINNECK,
        # "INVERTED HAMMER": ta.CDLINVERTEDHAMMER,
        # "KICKING": ta.CDLKICKING,
        # "KICKING BY LENGTH": ta.CDLKICKINGBYLENGTH,
        # "LADDER BOTTOM": ta.CDLLADDERBOTTOM,
        # "LONG LEGGED DOJI": ta.CDLLONGLEGGEDDOJI,
        # "LONG LINE": ta.CDLLONGLINE,
        # "MARUBOZU": ta.CDLMARUBOZU,
        # "MATCHING LOW": ta.CDLMATCHINGLOW,
        # "MAT HOLD": ta.CDLMATHOLD,
        # "MORNING DOJI STAR": ta.CDLMORNINGDOJISTAR,
        "MORNING STAR": ta.CDLMORNINGSTAR,
        # "ON NECK": ta.CDLONNECK,
        # "PIERCING": ta.CDLPIERCING,
        # "RICKSHAW MAN": ta.CDLRICKSHAWMAN,
        # "RISE FALL 3 METHODS": ta.CDLRISEFALL3METHODS,
        # "SEPARATING LINES": ta.CDLSEPARATINGLINES,
        "SHOOTING STAR": ta.CDLSHOOTINGSTAR,
        # "SHORT LINE": ta.CDLSHORTLINE,
        # "SPINNING TOP": ta.CDLSPINNINGTOP,
        # "STALLED PATTERN": ta.CDLSTALLEDPATTERN,
        # "STICK SANDWICH": ta.CDLSTICKSANDWICH,
        # "TAKURI": ta.CDLTAKURI,
        # "TASUKI GAP": ta.CDLTASUKIGAP,
        # "THRUSTING": ta.CDLTHRUSTING,
        # "TRISTAR": ta.CDLTRISTAR,
        # "UNIQUE 3 RIVER": ta.CDLUNIQUE3RIVER,
        # "UPSIDE GAP 2 CROWS": ta.CDLUPSIDEGAP2CROWS,
        # "XSIDE GAP 3 METHODS": ta.CDLXSIDEGAP3METHODS,
    }

    def informative_pairs(self):
        """
        - Get daily candles infor for all pairs
        """
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, "1d") for pair in pairs]
        return informative_pairs

    @informative("1d")
    def populate_indicators_1d(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        """
        -
        - price close > ema89  and (ema34 > ema89 or ema89 - ema34 < 5% of ema89)

        """
        # EMA
        dataframe["ema34"] = ta.EMA(dataframe, timeperiod=34)
        dataframe["ema89"] = ta.EMA(dataframe, timeperiod=89)

        # pattern recognition - bullish candlestick
        for (
            pattern_name,
            pattern_function,
        ) in self.bullish_candlestick_patterns.items():
            dataframe[pattern_name] = pattern_function(dataframe)
        return dataframe

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:

        # EMA
        dataframe["ema34"] = ta.EMA(dataframe, timeperiod=34)
        dataframe["ema89"] = ta.EMA(dataframe, timeperiod=89)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # pattern recognition - bullish candlestick
        for (
            pattern_name,
            pattern_function,
        ) in self.bullish_candlestick_patterns.items():
            dataframe[pattern_name] = pattern_function(dataframe)
        return dataframe

    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        """
        dataframe format
        - ema_trend: up, down
        - open, close, low, high
        - rsi_trend: up, down
        - price_high_trend: up, down
        - hammer_trend: up, down
        - candle_trend: up, down
        - enter_long: 1, 0

        """
        # Check daily is bullish
        # price close > ema89  and (ema34 > ema89 or ema89 - ema34 < 7% of ema89)
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema34_1d"])
                # & (
                #     (dataframe["ema34_1d"] > dataframe["ema89_1d"])
                #     | (
                #         (dataframe["ema89_1d"] - dataframe["ema34_1d"])
                #         < 0.1 * dataframe["ema89_1d"]
                #     )
                # )
            ),
            "daily_trend",
        ] = "up"

        # ema34 > ema89
        dataframe.loc[
            (dataframe["ema34"] >= dataframe["ema89"]), "ema_trend"
        ] = "up"
        dataframe.loc[
            (dataframe["ema34"] < dataframe["ema89"]), "ema_trend"
        ] = "down"

        # angle of ema34 in 20 previous candles larger than 30 degree
        # dataframe["ema34_angle"] = qtpylib.angle(dataframe["ema34"], 20)
        # Calculate the angle of ema34 over 20 periods
        dataframe["ema34_angle"] = calculate_angle(dataframe["ema34"], 20)
        dataframe.loc[(dataframe["ema34_angle"] > 30), "ema34_angle_trend"] = (
            "up"
        )

        # RSI
        dataframe.loc[
            (dataframe["rsi"] > 50) & (dataframe["rsi"] < 65), "rsi_trend"
        ] = "up"
        dataframe.loc[
            (dataframe["rsi"] <= 50) | (dataframe["rsi"] >= 65), "rsi_trend"
        ] = "down"

        # close price > open price
        dataframe.loc[
            (dataframe["close"] > dataframe["open"]), "price_high_trend"
        ] = "up"

        # Close price > EMA 34 and distance from low to EMA 34 < distance from open to low
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema34"])
                & (
                    (dataframe["low"] - dataframe["ema34"])
                    < (dataframe["open"] - dataframe["low"])
                )
                # & (
                #     (
                #         (dataframe["high"] - dataframe["close"])
                #         < (dataframe["open"] - dataframe["low"])
                #     )
                #     or (
                #         (dataframe["high"] - dataframe["close"])
                #         < (dataframe["close"] - dataframe["open"])
                #     )
                # )
            ),
            "hammer_trend",
        ] = "up"

        # Bullish candlestick pattern
        dataframe.loc[
            dataframe[list(self.bullish_candlestick_patterns.keys())]
            .gt(0)
            .any(axis=1),
            "candle_trend",
        ] = "up"

        # combine all trend -> signal
        dataframe.loc[
            (dataframe["ema_trend"] == "up")
            & (dataframe["rsi_trend"] == "up")
            & (dataframe["price_high_trend"] == "up")
            & (dataframe["hammer_trend"] == "up")
            & (dataframe["daily_trend"] == "up")
            & (dataframe["candle_trend"] == "up"),
            "enter_long",
        ] = 1

        # notify signal if previsous candle have enter_long = 1
        if dataframe["enter_long"].iloc[-1] == 1:
            # self.dp.send_msg(
            #     f"[H4] Signal: Enter Long for {metadata['pair']} with previous close price {dataframe['close'].iloc[-1]}"
            # )
            # get candle names for notification for current candle
            candle_names = []
            for pattern_name in self.bullish_candlestick_patterns.keys():
                if dataframe[pattern_name].iloc[-1] > 0:
                    candle_names.append(pattern_name)

            # Get current time in timezone HoChiMinh
            current_time = datetime.datetime.now(
                tz=pytz.timezone("Asia/Ho_Chi_Minh")
            ).strftime("%Y-%m-%d %H:%M:%S")
            self.dp.send_msg(
                f"H4 - SonicR - {current_time}: Enter Long above price {dataframe['close'].iloc[-1]} for {metadata['pair']} with {', '.join(candle_names)} candles"
            )
        return dataframe

    def populate_exit_trend(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        dataframe.loc[
            (
                # RSI crosse above 80
                dataframe["ema_trend"]
                == "down"
            ),
            "exit_long",
        ] = 1
        return dataframe
