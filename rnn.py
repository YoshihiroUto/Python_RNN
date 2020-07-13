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
import tensorflow as tf


# 映画レビューデータの読み込み
df = pd.read_csv('movie_data.csv')

print(df)


# *********************************************************************************************************************************
# データの前処理
# *********************************************************************************************************************************

'''
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
word_counts = sorted(counts, key=counts.get, reverse=True) # 大量の一意な単語のリストが入る
word_to_int = {word: ii for ii, word in enumerate(word_counts, 1)}
print(word_to_int)

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
y_train = df.loc[:25000, 'sentiment'].values
X_test  = sequences[25000:, :]
y_test  = df.loc[25000:, 'sentiment'].values

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





# *********************************************************************************************************************************
# RNNモデルの構築[ 以下の4つのメソッドを持つクラス ]
# 1：コンストラクタ
#   モデルのパラメータをすべて設定した後，計算グラフを作成し，self.buildメソッドを呼び出して多層RNNモデルを構築する
# 2：buildメソッド
#   入力データ・入力ラベル・隠れそうのドロップアウト設定のキープ率に対応する3つのプレースホルダを宣言．
#   宣言後，埋め込み層を作成し，埋め込み表現を入力として，多層RNNを構築する
# 3：trainメソッド
#   計算グラフを起動するためのTensorFlowセッションを作成し，計算グラフで定義されたコスト関数を最小化する．
#   ミニバッチを順番に処理しつつ，指定された数のエポックでトレーニングを行う．
#   チェックポイントとして，10エポック後のモデルを保存する
# 4：predictメソッド
#   新しいセッションを作成し，トレーニングプロセスで保存しておいた最後のチェックポイントを復元
#   テストデータで予測値を生成する
# *********************************************************************************************************************************
'''
1：コンストラクタ
'''
class SentimentRNN(object):
    def __init__(self, n_words, seq_len=200, lstm_size=256, num_layers=1, batch_size=64, learning_rate=0.0001, embed_size=200):
        self.n_words = n_words # 一意な単語の個数
        self.seq_len = seq_len # 入力するベクトルの次元数[時間刻みの総数]
        self.lstm_size = lstm_size # 各層の隠れユニットの個数
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embed_size = embed_size

        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(123)
            self.build()
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()

    '''
    2：buildメソッド
    '''
    def build(self):
        # プレースホルダを定義
        tf_x = tf.placeholder(tf.int32, shape=(self.batch_size, self.seq_len), name='tf_x')
        tf_y = tf.placeholder(tf.float32, shape=(self.batch_size), name='tf_y')
        tf_keepprob = tf.placeholder(tf.float32, name='tf_keepprob')

        # 埋め込み層を作成
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
        　今回は，上記のマッピング作業を，TensorFlowのtf.nn.embedding_lookup関数がやってくれる
        '''
        # 1.サイズが　一意な単語の個数　×　埋め込み先のベクトル次元数　の行列をembeddingというテンソル変数[TensorFlow用の変数]として作成
        #                  n_words             embedding_size
        #   この行列の要素を，-1~1の変数で初期化する．
        embedding = tf.Variable(tf.random_uniform((self.n_words, self.embed_size), minval=-1, maxval=1), name='embeded_x')
        # 2. tf.nn.embedding_lookup関数を呼び出し，tf_x[入力層]の各要素に関連する埋め込み行列の行を特定する
        #    渡しているのは埋め込みテンソルと検索IDの2つ．
        embed_x = tf.nn.embedding_lookup(embedding, tf_x, name='embeded_x')

        # LSTMセルを定義し，積み上げる
        cells = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(self.lstm_size),
                output_keep_prob=tf_keepprob)
            for i in range(self.num_layers)])
        
        # 初期状態を定義
        self.initial_state = cells.zero_state(self.batch_size, tf.float32)
        print(' << initial state >> ', self.initial_state)

        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cells, embed_x, initial_state=self.initial_state)

        # 注意：lstm_outputsの形状：[batch_size, max_time, cells.output_size]
        print('\n << lstm_output >> ', lstm_outputs)
        print('\n << final state >> ', self.final_state)

        # RNNの出力の後に全結合層を適用
        logits = tf.layers.dense(inputs=lstm_outputs[:, -1], units=1, activation=None, name='logits')

        logits = tf.squeeze(logits, name='logits_squeezed')
        print('\n << logits >> ', logits)

        y_proba = tf.nn.sigmoid(logits, name='probabilities')
        predictions = {
            'probabilities': y_proba,
            'labels' : tf.cast(tf.round(y_proba), tf.int32, name='labels')
        }
        print('\n << predictions >> ', predictions)

        # コスト関数を定義
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y,logits=logits), name='cost')

        # オプティマイザを定義
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(cost, name='train_op')

        print('(; ･`д･´)embed_x')
        print(embed_x)

    '''
    3：trainメソッド
    '''
    def train(self, X_train, y_train, num_epochs):
        with tf.Session(graph=self.g) as sess:
            sess.run(self.init_op)
            iteration = 1
            for epoch in range(num_epochs):
                state = sess.run(self.initial_state)

                for batch_x, batch_y in create_batch_generator(X_train, y_train, self.batch_size):
                    feed = {'tf_x:0' : batch_x,
                            'tf_y:0' : batch_y,
                            'tf_keepprob:0' : 0.5,
                            self.initial_state: state}
                    loss, _, state = sess.run(
                        ['cost:0', 'train_op', self.final_state],
                        feed_dict=feed)
                    if iteration % 20 == 0:
                        print("Epoch: %d/%d Iteration: %d | Train loss: %.5f" % (epoch + 1, num_epochs, iteration, loss))
                    iteration += 1
                if (epoch+1)%10 == 0:
                    self.saver.save(sess, "model/sentiment-%d.ckpt" % epoch)

    '''
    4：predictメソッド
    '''
    def predict(self, X_data, return_proba=False):
        preds = []
        with tf.Session(graph = self.g) as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('./model/'))
            test_state = sess.run(self.initial_state)
            for ii, batch_x in enumerate(create_batch_generator(X_data, None, batch_size=self.batch_size), 1):
                feed = {'tf_x:0' : batch_x, 'tf_keepprob:0': 1.0, self.initial_state: test_state}
                if return_proba:
                    pred, test_state = sess.run(
                        ['probabilities:0', self.final_state],
                        feed_dict=feed)
                else:
                    pred, test_state = sess.run(
                        ['labels:0', self.final_state],
                        feed_dict=feed)
                preds.append(pred)
            return np.concatenate(preds)



n_words = max(list(word_to_int.values())) + 1

# 変数説明
# ⇒ n_words:一意な単語の総数
# ⇒ sequence_length
# ⇒
# ⇒
# ⇒
# ⇒
# ⇒
# ⇒
# ⇒

rnn = SentimentRNN(n_words=n_words, seq_len=sequence_length, embed_size=256, lstm_size=128, num_layers=1, batch_size=100, learning_rate=0.001)

rnn.train(X_train, y_train, num_epochs=40)

preds = rnn.predict(X_test)
y_true = y_test[:len(preds)]
print('Test Acc.: %.3f' %( np.sum(preds == y_true) / len(y_true) ))