## old features 

    we have at start 6 features :
    open         → price when day STARTED
               "ETH opened at $3,200 today"

    high         → highest price reached DURING the day
               "ETH touched $3,350 at some point today"

    low          → lowest price reached DURING the day
               "ETH dropped to $3,150 at some point today"

    close        → price when day ENDED
               "ETH closed at $3,280 tonight"

    volume       → how many COINS were traded today on Binance
               buy + sell combined
               "500,000 ETH coins changed hands today"

    quote_volume → how much USD was exchanged today on Binance
               buy + sell combined in dollar value
               "buyers spent + sellers gained = $1,640,000,000"

## some trading words and visulization
    candle body:
    green: buyers were stronger , close price high (bullish)
    red : seller were stronger , close price is down (bearish)


    body:
    big-> means that stroong market move 
    small-> weak move and buyers and seller is same


    Long top wick → price tried to go high but got rejected back down
    Long bottom wick → price tried to go low but got pushed back up
    Long wicks = uncertainty in the market


    MA_7 vs MA_30  = compare short term vs long term
                    MA_7 > MA_30 = Golden Cross → price going up 
                    MA_7 < MA_30 = Death Cross  → price going down 

# new features-feature_engeering:
    candle_body    = close - open
                 "did today end higher or lower than it started?"
                 positive = green day (buyers stronger)
                 negative = red day (sellers stronger)

    high_low_range = high - low
                    "how wild was today?"
                    big number   = volatile day
                    small number = calm day   (notice!! the big number depend on coin like BTC if range is 2000 normall 
                    smth like dodge not normal at all i think it is never happen)

    MA_7           = average close of last 7 days (short term)
                    think of it as "what happened this week"

    MA_30          = average close of last 30 days (long term)
                    "what is the long term trend?"
                    think of it as "what happened this month"

    moving_1d      = (today_close - yesterday_close) / yesterday_close × 100
                    "how much % did price move today?"
                    +3%  = price rose 3% today
                    -5%  = price dropped 5% today

    volatility_7   = standard deviation of return_1d last 7 days
                    "is market nervous or calm lately?"
                    high = jumping around a lot
                    low  = stable and predictable

                    (5% to 10% very high volatility)

    volumelast7(ratio)  = today_volume / avg_volume_last_7_days
                    "is today's trading unusual?"
                    2.0  = twice as active as normal
                    0.5  = half as active as normal

    z_score        = (close - rolling_mean_30) / rolling_std_30
                    "is today's price acting weird?"
                    near 0     = normal, nothing unusual
                    +2 or -2   = price is unusually high or low

    tomorrow_close (dropped ) = get tomorrow close
    
    tomorrow_return (dropped) = get percentage how much today price is different from tomorrow

    (notice!!: this two columns made to just make the target as we make label if it more 3 or less -3 or between them)

    target (Y) = we have 3 categories (Bigup , Bigdown , Stable)  threshold we are using 3% for tomorrow return as we try to find balance in classes

    (notice!!: there is a class imbalance between classes but we will not deducte the 3% as it will be very normal day if we just make 2% so thats not real for realife using , or making 2 classes binary like moving , not moving as model will predict high but it will not be used in real life  , when we start modeling we gonna use smote that will generate fake data for model or class_weight =balanced )

