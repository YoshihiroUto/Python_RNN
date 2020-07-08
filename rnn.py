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

print(df)

'''
データの前処理；
単語の分割をし，各単語の出現回数をカウント
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

print(df)

'''
マッピングを作成：
一意な単語をそれぞれ整数にマッピング．
(映画レビューのテキスト全体を数値のリストに変換するためのプログラム)
    mapped_reviews -> テキストデータを数値データに変換したとのデータ構造が代入される
'''
word_counts = sorted(counts, key=counts.get, reverse=True)
print(word_counts[:5])
word_to_int = {word: ii for ii, word in enumerate(word_counts, 1)}

mapped_reviews = []
pbar = pyprind.ProgBar(len(df['review']), title='Map reviews to ints')
for review in df['review']:
    mapped_reviews.append([word_to_int[word] for word in review.split()])
    pbar.update()



