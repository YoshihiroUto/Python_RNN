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


'''
各テキストデータのシーケンスを同じ長さに統一させる：
・シーケンスの長さが200単語未満の場合，左側を0でパディング
・シーケンスの長さが200単語以上の場合，最後の200単語の要素を使用する
    sequence_length -> 時間分解能が200単語ということを表し，これらを変化させることでパフォーマンスの変化を調査することも面白いかも
'''
sequence_length = 200
# とりあえず作成後のデータセットを0で初期化しておく
sequences = np.zeros((len(mapped_reviews), sequence_length), dtype=int)

for i, row in enumerate(mapped_reviews):
    review_arr = np.array(row)
    # 右詰めで値を上書きしていく
    sequences[i, -len(row):] = review_arr[-sequence_length:]

print(sequences[:20,:])


'''
トレーニングセットとテストセットの分割
'''
X_train = sequences[:25000, :]
Y_train = df.loc[:25000, 'sentiment'].values
X_test  = sequences[25000:, :]
Y_test  = df.loc[25000:, 'sentiment'].values

''' Helper[ ミニバッチを生成する ]
与えられたデータセット(トレーニングセットorテストセット)をチャンクに分割し，
これらのチャンクを反復的に処理するためのジェネレータを返す．
'''
np.random.seed(123) # 乱数を再現可能にするため

# ミニバッチを生成する関数を定義
def create_batch_generator(x, y=None, batch_size=64):
    n_batches = len(x) // batch_size
    x = x[:n_batches*batch_size]
    if y is not None:
        y = y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        if y is not None:
            yield x[ii:ii+batch_size], y[ii:ii+batch_size]
        else:
            yield x[ii:ii+batch_size]

'''埋め込み[ embedding ]
埋め込み = 文や単語、文字など自然言語の構成要素に対して、何らかの空間におけるベクトルを与えること
・one-hotエンコーディング
　インデックスを0と1のベクトルに変換すること．
　たとえば、映画レビューデータのすべてにおいて出現する一意な単語の総数を20000単語とする．
　まず20000次元のベクトルを作成し，20000単語のうち，各テキストデータが含んでいる単語に対応する次元のみ1という値が入り，それ以外を0とする方法
　※次元数がめちゃんこ多くなってしまう．
　※しかも0か1しかないベクトルなので，特徴量的に考えてめちゃ疎

上記の問題を解決するための手法⇓
・埋め込み
　各単語を実数地の要素を持つ固定サイズのベクトルにマッピングする．  

'''