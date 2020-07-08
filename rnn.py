'''
必要なモジュールの読み込み
    pyprind -> 筆者が数年前に作成した，進行状態と推定終了時間を確認できるようにするためのライブラリ
    pandas  -> データ整形のための外部ライブラリ．超有名
    punctuation -> <=>?@[\]^_`{|}~.　こんな文字列
    re      -> 正規表現モジュール
    numpy   -> 数値計算を簡単にかけたりするようになるライブラリ
'''
import pyprind
import pandas as pd
from string import punctuation
import re
import numpy as np

# 映画レビューデータの読み込み
df = pd.read_csv('movie_data.csv')



