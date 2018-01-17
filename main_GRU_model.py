import os
import gc
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import datetime
import time

from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard
from keras import backend as K
from keras import optimizers
from keras import losses
from keras.preprocessing.sequence import pad_sequences
from keras import initializers

datafolder = '../input/'



def get_glove():
    glove_path = 'D:/Users/shanger_lin/Desktop/jupyter_working/NLP projects/embedding/glove.6B/glove.6B.50d.txt'
    with open(glove_path, 'r', encoding='utf8') as f:
        vocab = list()
        embedding = list()
        glove_vectors = dict()
        for line in f:
            splitline = line.split()
            word = splitline[0]
            vector = np.array([float(val) for val in splitline[1:]])

            vocab.append(word)
            glove_vectors[word] = vector
            embedding.append(vector)
        embeddings = np.array(embedding)
    return vocab, embeddings, glove_vectors


vocab, embeddings, embeddings_index = get_glove()

# word2token = {w: i for i, w in enumerate(vocab)}


start_time = time.time()

filenames = os.listdir(datafolder)

gc.collect()
data = dict()
for i in filenames:
    topic, *mid, filetype = i.split('.')
    if filetype == 'tsv':
        data[topic] = pd.read_table(datafolder + i)
    elif filetype == 'csv':
        data[topic] = pd.read_csv(datafolder + i)
    else:
        pass

# %%time
# sub_df = data['sample_submission']
# train_df = data['train']
# test_df = data['test']

# %%time
# train_df.rename(columns={'train_id': 'id'}, inplace=True)
# test_df.rename(columns={'test_id': 'id'}, inplace=True)
# train_df['is_train'] = 1
# test_df['is_train'] = 0
# df = pd.concat([train_df, test_df])

# %%time
# def fillna_with_other_column(df, to_fill='item_description', material='name'):
#     to_fill_isna = df[to_fill].isna()
#     df[to_fill][to_fill_isna] = df[material][to_fill_isna]
# fillna_with_other_column(df, to_fill='item_description', material='name')

# %%time
# def fillna_by_mode_of_shared_value_of_other_column(df, to_fill='category_name', material='name'):
#     category_na = df[to_fill].isna()
#     groups = [v for k, v in df.groupby(material).groups.items() if (
#         (len(v)>1) and
#         (df[to_fill].iloc[v].isna().sum()) and
#         (df[to_fill].iloc[v].notna().sum()))]

#     for i in groups:
#         df[to_fill].loc[category_na & df.index.isin(i)] = df.iloc[i][to_fill].dropna().mode().values[0]
# fillna_by_mode_of_shared_value_of_other_column(df, to_fill='category_name', material='name')

# %%time
# brands = train_df['brand_name'].dropna().drop_duplicates()
# brand_counter = CountVectorizer()
# brand_counter.fit(brands)
# brand_token = (brand_counter.transform(brands).T/(brands.apply(lambda x: len(x.split())).values))

# def fillna_catgory_from_other(df, to_fill='brand_name', material='name'):
#     brandna = df[to_fill].isna()
#     name_trans = brand_counter.transform(df[material][brandna].values)
#     na_brands = list()
#     for i, r in enumerate(name_trans):
#         try:
#             na_brands.append(brands.iloc[np.where(np.dot(r.toarray().reshape(1, -1), brand_token)==1)[1][0]])
#         except IndexError:
#             na_brands.append(np.NAN)
#     df[to_fill][brandna] = na_brands
# fillna_catgory_from_other(df, to_fill='brand_name', material='name')


sub_df = data['sample_submission']
train_df = data['train']
test_df = data['test']


df = pd.read_csv('combined_na_filled_v2.csv')


filled_train_df = df[df['is_train'] == 1]
del filled_train_df['is_train']
filled_test_df = df[df['is_train'] == 0]
del filled_test_df['is_train']


to_fill_cols = ['brand_name', 'category_name', 'item_description']
to_fill_col = []
for col in to_fill_cols:
    if np.random.ranf(1)[0] > 0.5:
        train_df[col] = filled_train_df[col]
        to_fill_col.append(col)

if not to_fill_col:
    to_fill_col = 'None'
else:
    to_fill_col = '_'.join(to_fill_col)
print('to_fill_col', to_fill_col)
# train_df[to_fill_col] = filled_train_df[to_fill_col]


train = train_df
test = test_df


# train['target'] = np.log1p(train['price'])
train['target'] = train['price']

print(train.shape)
print(test.shape)
print('5 folds scaling the test_df')
test_len = test.shape[0]


def simulate_test(test):
    if test.shape[0] < 800000:
        indices = np.random.choice(test.index.values, 2800000)
        test_ = pd.concat([test, test.iloc[indices]], axis=0)
        return test_.copy()
    else:
        return test


test = simulate_test(test)
print('new shape ', test.shape)
print('[{}] Finished scaling test set...'.format(time.time() - start_time))


# HANDLE MISSING VALUES
print("Handling missing values...")


def handle_missing(dataset):
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return dataset


train = handle_missing(train)
test = handle_missing(test)
print(train.shape)
print(test.shape)

print('[{}] Finished handling missing data...'.format(time.time() - start_time))


# PROCESS CATEGORICAL DATA


print("Handling categorical variables...")
le = LabelEncoder()

le.fit(np.hstack([train.category_name, test.category_name]))
train['category'] = le.transform(train.category_name)
test['category'] = le.transform(test.category_name)

le.fit(np.hstack([train.brand_name, test.brand_name]))
train['brand'] = le.transform(train.brand_name)
test['brand'] = le.transform(test.brand_name)
del le, train['brand_name'], test['brand_name']

print('[{}] Finished PROCESSING CATEGORICAL DATA...'.format(time.time() - start_time))


# PROCESS TEXT: RAW
print("Text to seq process...")
print("   Fitting tokenizer...")


raw_text = np.hstack([train.category_name.str.lower(),
                      train.item_description.str.lower(),
                      train.name.str.lower()])


tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)
word_index = tok_raw.word_index

EMBEDDING_DIM = 50  # better way to show?
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


print("   Transforming text to seq...")
train["seq_category_name"] = tok_raw.texts_to_sequences(train.category_name.str.lower())
test["seq_category_name"] = tok_raw.texts_to_sequences(test.category_name.str.lower())
train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description.str.lower())
test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description.str.lower())
train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())

print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))


# EXTRACT DEVELOPTMENT TEST


dtrain, dvalid = train_test_split(train, train_size=0.99)
# dtrain, dvalid = train_test_split(train, train_size=0.99)
# dtrain, dvalid = train, train
print(dtrain.shape)
print(dvalid.shape)


# EMBEDDINGS MAX VALUE
# Base on the histograms, we select the next lengths
MAX_NAME_SEQ = 20  # 17
MAX_ITEM_DESC_SEQ = 60  # 269
MAX_CATEGORY_NAME_SEQ = 20  # 8
MAX_TEXT = np.max([np.max(train.seq_name.max()),
                   np.max(test.seq_name.max()),
                   np.max(train.seq_category_name.max()),
                   np.max(test.seq_category_name.max()),
                   np.max(train.seq_item_description.max()),
                   np.max(test.seq_item_description.max())]) + 2
MAX_CATEGORY = np.max([train.category.max(), test.category.max()]) + 1
MAX_BRAND = np.max([train.brand.max(), test.brand.max()]) + 1
MAX_CONDITION = np.max([train.item_condition_id.max(),
                        test.item_condition_id.max()]) + 1

print('[{}] Finished EMBEDDINGS MAX VALUE...'.format(time.time() - start_time))


# KERAS DATA DEFINITION
def get_keras_data(dataset):
    x = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand': np.array(dataset.brand),
        'category': np.array(dataset.category),
        'category_name': pad_sequences(dataset.seq_category_name, maxlen=MAX_CATEGORY_NAME_SEQ),
        'item_condition': np.array(dataset.item_condition_id),
        'shipping': np.array(dataset[["shipping"]])
    }
    return x


X_train = get_keras_data(dtrain)
X_valid = get_keras_data(dvalid)
X_test = get_keras_data(test)

print('[{}] Finished DATA PREPARARTION...'.format(time.time() - start_time))


gc.collect()


def get_model():

    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand = Input(shape=[1], name="brand")
    category = Input(shape=[1], name="category")
    category_name = Input(shape=[X_train["category_name"].shape[1]],
                          name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    shipping = Input(shape=[X_train["shipping"].shape[1]], name="shipping")

    # Embeddings layers
    emb_size = EMBEDDING_DIM

    vocab_size = embedding_matrix.shape[0]
    GLOVE_TRAINABLE = False
    glove_name = Embedding(vocab_size, emb_size, trainable=GLOVE_TRAINABLE, weights=[embedding_matrix])(name)
    glove_item_desc = Embedding(vocab_size, emb_size, trainable=GLOVE_TRAINABLE, weights=[embedding_matrix])(item_desc)
    glove_category_name = Embedding(vocab_size, emb_size, trainable=GLOVE_TRAINABLE, weights=[embedding_matrix])(category_name)

    emb_name = Embedding(vocab_size, 20)(name)
    emb_item_desc = Embedding(vocab_size, 60)(item_desc)
    emb_category_name = Embedding(vocab_size, 20)(category_name)

    emb_brand = Embedding(MAX_BRAND, 10)(brand)
    emb_category = Embedding(MAX_CATEGORY, 10)(category)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)

    glove_item_desc_layer = GRU(16)(glove_item_desc)
    glove_category_name_layer = GRU(8)(glove_category_name)
    glove_name_layer = GRU(8)(glove_name)

    emb_item_desc_layer = GRU(16)(emb_item_desc)
    emb_category_name_layer = GRU(8)(emb_category_name)
    emb_name_layer = GRU(8)(emb_name)

    # main layer
    main_l = concatenate([
        Flatten()(emb_brand),
        Flatten()(emb_category),
        Flatten()(emb_item_condition),
        glove_item_desc_layer,
        glove_category_name_layer,
        glove_name_layer,
        emb_item_desc_layer,
        emb_category_name_layer,
        emb_name_layer,
        shipping,
    ])
    # main_l = Dropout(0.25)(Dense(1024, activation='relu')(main_l))
    main_l = Dropout(0.25)(Dense(512, activation='relu')(main_l))
    main_l = Dropout(0.2)(Dense(512, activation='relu')(main_l))
    main_l = Dropout(0.2)(Dense(64, activation='relu')(main_l))

    # output
    output = Dense(1, activation="linear")(main_l)

    # model
    model = Model([name,
                   item_desc,
                   brand,
                   category,
                   category_name,
                   item_condition,
                   shipping], output)
    # optimizer = optimizers.RMSprop()
    optimizer = optimizers.Adam() # lr=lr_init, decay=lr_decay
    model.compile(loss=losses.mean_squared_logarithmic_error,
                  optimizer=optimizer)
    return model


# fin_lr=init_lr * (1/(1+decay))**(steps-1)
# exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1

print('[{}] Finished DEFINEING MODEL...'.format(time.time() - start_time))


# FITTING THE MODEL
epochs = 6
BATCH_SIZE = 512 * 4
steps = int(len(X_train['name']) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.01, 0.0001
# lr_decay = exp_decay(lr_init, lr_fin, steps)
lr_decay = (lr_init - lr_fin) / steps
log_subdir = '_'.join(['ep', str(epochs),
                       'bs', str(BATCH_SIZE),
                       'lrI', str(lr_init),
                       'lrF', str(lr_fin),
                       'dr', str(lr_decay)])

model = get_model()

# K.set_value(model.optimizer.lr, lr_init)
# K.set_value(model.optimizer.decay, lr_decay)


earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
history = model.fit(X_train, dtrain.target,
                    epochs=epochs,
                    batch_size=BATCH_SIZE,
                    validation_split=0.01,
                    verbose=1,
                    shuffle=True,
                    callbacks=[earlyStopping],
                    )
print('[{}] Finished FITTING MODEL...'.format(time.time() - start_time))
# EVLUEATE THE MODEL ON DEV TEST

gc.collect()

train_preds = model.predict(X_valid, batch_size=BATCH_SIZE)
v_rmsle = np.sqrt(mean_squared_error(np.log1p(train_preds), np.log1p(dvalid.target)))
print('v_rmsle', v_rmsle)


# CREATE PREDICTIONS
# preds = np.expm1(model.predict(X_test, batch_size=BATCH_SIZE))
preds = model.predict(X_test, batch_size=BATCH_SIZE)

test.rename(columns={'id': 'test_id'}, inplace=True)
submission = test[["test_id"]][:test_len]
submission["price"] = preds[:test_len]

print('[{}] Finished predicting test set...'.format(time.time() - start_time))


now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

message = 'use_default_lr'


submit_name = "{}_GRU_{}_batch{}_lr{}_{}_{}.csv".format(
    now,
    to_fill_col,
    BATCH_SIZE,
    ''.join(map(str, [lr_init, lr_fin])),
    v_rmsle,
    message)

submission.to_csv('_results/{}'.format(submit_name), index=False)
print('result saved as : {}'.format(submit_name))
