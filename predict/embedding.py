import pandas as pd
import numpy as np
import ast
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

word2vec = Word2Vec.load("predict/data/word2vec.model")

df = pd.read_csv('predict/data/merged_df_최종.csv', index_col='Unnamed: 0')
# sim_sorted_ind = np.load('predict/data/sim_sorted_after.npy', allow_pickle=True)

mean_vector = np.load('predict/data/mean_vector.npy')

w2v_sim = cosine_similarity(mean_vector, mean_vector)
sim_sorted_ind=w2v_sim.argsort()[:,::-1]

# 단어 벡터 평균구하기
def vectors(embedding):
    # vector size 만큼
    tmp = np.zeros(50)
    count = 0

    # embedding = list of list corpus
    for word in embedding:
        try:
            # word에 해당하는 단어를 워드투백에서 찾아 해당 벡터를 리스트에 붙힘
            # 100차원으로 형성됨
            tmp += word2vec.wv[word]
            count += 1
        except:
            pass

    tmp /= count  # 아이템 갯수로 전체 벡터를 mean해줌

    return tmp


def aggregate_vectors(string_list):
    product_vec = []
    for noun in string_list:
        try:
            product_vec.append(word2vec.wv[noun])
        except KeyError:
            continue
    vector = np.mean(product_vec, axis=0)

    return vector



def make_list(string):
    try :
        return ast.literal_eval(string)
    except :
        return list()



def age_col(age):
    if age <= 18:
        a = "age18"
    elif (age > 18) & (age <= 23):
        a = "age19_23"
    elif (age > 23) & (age <= 28):
        a = "age24_28"
    elif (age > 28) & (age <= 33):
        a = "age29_33"
    elif (age > 33) & (age <= 39):
        a = "age34_39"
    else:
        a = "age40"
    return a

def gender_col(gender):
    if gender == "M":
        a = "man"
    else:
        a = "woman"
    return a

# 키워드 유사도 높은 순으로 상품 리스트 뽑아낸다
def index_out(keyword, df):
    word = df["word"].apply(lambda x: make_list(x)).tolist()

    vec_input = aggregate_vectors(keyword)

    # load
    mean_vector = np.load('predict/data/mean_vector.npy')

    cosine_sim = []
    for idx, vec in enumerate(mean_vector):
        vec1 = vec_input.reshape(1, -1)
        vec2 = vec.reshape(1, -1)
        cos_sim = cosine_similarity(vec1, vec2)[0][0]
        cosine_sim.append((idx, cos_sim))

    # 첫번째 상품부터의 코사인 유사도
    temp_sim = []
    for elem in cosine_sim:
        temp_sim.append(elem[1])

    cosine_sim.sort(key=lambda x: -x[1])

    # 키워드 유사도 높은 순으로 상품 리스트 뽑아낸다
    # sim_sorted_ind = 아이템 * 아이템 유사도 높은 순으로 인덱스가 정렬되어 있는 matrix
    li = sim_sorted_ind[cosine_sim[0][0]]

    return li, temp_sim


def item_filtering(main_category, coordi, sim_df):
    filtered = sim_df[sim_df["main_category"].isin(main_category)].reset_index(drop=True)

    if coordi == [""]:  # coordi 값 입력 안해도 값 나오게
        filtered_both = filtered
    else:
        a = []
        for i in range(filtered.shape[0]):
            inter = list(set(ast.literal_eval(filtered.loc[i, "coordi"])) & set(coordi))
            if len(inter) >= 1:
                a.append(i)
        filtered_index = pd.DataFrame(index=a).index
        filtered_both = filtered.loc[filtered_index,].reset_index(drop=True)

    return filtered_both


def recsys(main_category, coordi, keyword):
    li, temp_sim = index_out(keyword, df)

    # 유사도 열 추가
    df['wv_cosine'] = temp_sim

    temp = df[df['word_string'].str.contains(keyword)]

    name_list = df.loc[li, 'name'].to_list()

    sim_df = temp[temp['name'].isin(name_list)]
    sim_df = sim_df.reset_index(drop=True)

    recsys_df = item_filtering(main_category, coordi, sim_df)

    recsys_df = recsys_df.sort_values(by=["wv_cosine"], ascending=False)

    return recsys_df



################ 이미지 처리 feature extraction #######################

##이미지 featur를 저장한 csv파일을 불러오는 함수
def load_img_feature():
    img_feature = pd.read_csv("predict/data/img_feature.csv", index_col=0)
    return img_feature


img_df = load_img_feature()

## 이미지 유사도로 검색 - 총 100개의 유사 제품 리스트 return
# input data: 검색 시 텍스트 유사도 검색 시 가장 관련 있는 제품명
# 유사도 측정 방식: 유클리디안 거리: no1(input data의 feature)과 items(no1을 포함한 모든 아이템의 feature) 사이의 거리 측정
# output data: 이미지 기준 input data와 가장 유사한 제품 100개 리스트 (제품명만)

def img_sim(img_feature, name):
    ## L2 norm 방식으로 유사도 측정
    # input data name의 feature 불러오기
    no1 = img_feature.loc[img_feature['name'] == name, "0":"255"].values
    items = img_feature.loc[:, "0":"255"].values

    # 이미지 유사도 거리 계산
    dists = np.linalg.norm(items - no1, axis=1)

    # 유클리디안 거리가 가장 가까운 2000개의 상품명 리스트 추출
    idxs = np.argsort(dists)[:2000]
    scores = [img_feature.loc[idx, "name"] for idx in idxs]

    return scores


############################
# 상품 중복 제거
def remove_dupe_dicts(l):
    return [dict(t) for t in {tuple(d.items()) for d in l}]


def wordcloud(wc_df):
    #### Wordcloud 만들기
    from wordcloud import WordCloud
    from collections import Counter
    string_list = wc_df['review_tagged_cleaned']

    try:
        string_list = string_list.apply(lambda x: ast.literal_eval(x))
    except:
        pass

    word_list = []
    for words in string_list:
        for word in words:
            if len(word) > 1:
                word_list.append(word)
    # 가장 많이 나온 단어부터 40개를 저장한다.
    counts = Counter(word_list)
    tags = counts.most_common(20)
    font = 'static/fonts/NanumSquareL.otf'

    word_cloud = WordCloud(font_path=font, background_color='black', max_font_size=400,
                           colormap='prism').generate_from_frequencies(dict(tags))
    word_cloud.to_file('static/무신사.png')
    print('wordcloud 완료')
    # 사이즈 설정 및 화면에 출력
    ####