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


'''
データの前処理；
# 単語の分割をし，各単語の出現回数をカウント
'''
# 大量のデータセットから一意の単語を見つけるのに便利なcollectionsパッケージのCounterを読み込む
from collections import Counter

# Counterクラスからcountsオブジェクトを作成
counts = Counter()
# データの入力数でプログレスバーを初期化[終了時間計算用？]
pbar = pyprind.ProgBar(len(df['review']), title='Counting words occurrences')
for i,review in enumerate(df['review']):
    text = ''.join([c if c not in punctuation else ' '+c+' ' for c in review]).lower()
    df.loc()[i, 'review'] = text
    pbar.update()
    counts.update(text.split())

