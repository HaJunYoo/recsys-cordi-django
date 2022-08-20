from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
import pandas as pd
import numpy as np
import ast
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from predict.models import PredResults

from predict.embedding import *
from predict.list import *

from timeit import default_timer as timer

from konlpy.tag import Okt

tokenizer = Okt()

##이미지 featur를 저장한 csv파일을 불러오는 함수
def load_img_feature():
    img_feature = pd.read_csv("predict/data/img_feature.csv", index_col=0)
    return img_feature

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

#############################################################
# 데이터 선언

#   코사인유사도를 구한 행렬을 역순으로 정렬 -> 유사도가 높은 순의 인덱스로 정렬됨
#   시간 복잡도가 제일 오래 걸림 => 여기서 시간을 제일 많이 소모됨 => O(n*logn)
#   sim_sorted_ind=sim.argsort()[:,::-1]
sim_sorted_ind = np.load('predict/data/sim_sorted_after.npy')
df = pd.read_csv('predict/data/merged_df_최종.csv', index_col='Unnamed: 0')
img_df = load_img_feature()

################################################################
# embeddings.py에서 함수들 임포트

###################################################

df["tags"] =  df["tags"].apply(lambda x : make_list(x))
df["review_tagged_cleaned"] =  df["review_tagged_cleaned"].apply(lambda x : make_list(x))
df["coordi"] =  df["coordi"].apply(lambda x : make_list(x))

######## mean_vector.npy 불러오는 것으로 아래 코드 해결

# topic modeling
# embedding.py

############################
# 상품 중복 제거
def remove_dupe_dicts(l):
  return [dict(t) for t in {tuple(d.items()) for d in l}]


def wordcloud(wc_df, wordcloud=None):
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

# view 함수
def predict(request):

    if request.POST.get('action') == 'post':

        # Receive data from client(input)
        # gender = str(request.POST.get('gender'))
        # age = int(request.POST.get('age'))
        main_category = str(request.POST.get('main_category'))
        coordi = str(request.POST.get('coordi'))
        input_text = str(request.POST.get('input_text'))
        top_n = int(request.POST.get('topn'))

        # 가방,모자,상의 <= 이런 양식으로 받아온다.
        print(main_category)
        print(coordi)

        # coordi, category list 화
        main_category = main_category.split(",")
        coordi = coordi.split(",")


        print('카테고리 :', main_category)
        print('코디 :',coordi)
        print('인풋 텍스트 :',input_text)
        print('topn : ', top_n)

        # Make prediction
        try :
            tot_result = recsys(main_category, coordi, input_text)

            result = tot_result[:top_n]
            result = result.sort_values(by=["wv_cosine", "scaled_rating", "year"], ascending=False)

            print('1단계 :', result)
            print('2단계 :',result.columns)

            classification = result[['name', 'img', 'review', 'price']]
            name = list(classification['name'])
            img = list(classification['img'])
            review = list(classification['review'])
            price = list(classification['price'])
            print(name)

            records = PredResults.objects.all()
            records.delete()

            for i in range(len(classification)):
                PredResults.objects.create(name=name[i], img=img[i], review=review[i] ,price=price[i])

            print('DB 저장 완료')

            try :
                wordcloud(result)
            except :
                pass

            return JsonResponse({'name': name, 'img': img}, safe=False)

        except :
            return JsonResponse({'name': "해당되는 추천이 없습니다. 다시 입력해주세요"}, safe=False)

    else :
        return render(request, 'predict.html', {'category_list': category_list, 'coordi_list': coordi_list})


# image classification view 함수
def img_predict(request):
    if request.POST.get('action') == 'post':

        input_text = str(request.POST.get('input_text'))
        top_n = int(request.POST.get('topn'))
        main_category = str(request.POST.get('main_category'))

        print(input_text , top_n, main_category)
        # Make prediction
        try :
            start = timer()
            # Okt가 시간이 엄청 걸림

            string_list = word_tokenize(input_text) # 리스트 형태
            #
            vec_input = aggregate_vectors(string_list)
            # vec_input = word2vec.wv[input_text]
            print('1단계', vec_input[:10])

            mean_vector = np.load('predict/data/mean_vector.npy')

            # 인풋 임베딩 단어와 모든 상품간의 유사도 => 열벡터 생성
            cosine_sim = []
            for idx, vec in enumerate(mean_vector):
                vec1 = vec_input.reshape(1, -1)
                vec2 = vec.reshape(1, -1)
                cos_sim = cosine_similarity(vec1, vec2)[0][0]
                cosine_sim.append((idx, cos_sim))
            print('2단계', cosine_sim[:10])

            temp_sim = []
            for elem in cosine_sim:
                temp_sim.append(elem[1])

            temp_df = df.copy()
            temp_df['wv_cosine'] = temp_sim

            # 만약 인풋 키워드가 word string 안에 포함이 되어 있다면 수행하고 없을 경우 그냥 기존대로 예외 처리

            # if temp_df['name'].str.contains(input_text):
            #
            #     sim_df = temp_df[temp_df['name'].str.contains(input_text)]
            #     print('3단계', sim_df)
            #     name = sim_df['name'].values[0]
            #
            #     print('4단계', name)
            #     # 상위 1개의 그림과 비교
            #     img_score = img_sim(img_df, name)
            #
            #     print('5단계', img_score[:10])
            #     # 이미지 유사도
            #     temp = temp_df[temp_df['name'].isin(img_score)]
            #     print('6단계', temp)
            #     print('7단계', temp.columns)
            #     temp = temp[:top_n]
            #     temp = temp.sort_values(by='scaled_rating', ascending=False)
            #     # print(temp.columns)
            #     print('9단계', temp)


            # else :
            sim_df = temp_df[temp_df['main_category'].str.contains(main_category)]

            print('3단계', sim_df)

            name1 = sim_df['name'].values[0]
            name2 = sim_df['name'].values[1]
            name3 = sim_df['name'].values[2]

            print('4단계', name1, name2, name3)
            # 상위 3개의 그림과 비교
            img_score1 = img_sim(img_df, name1)  # 2000개 추출
            img_score2 = img_sim(img_df, name2)
            img_score3 = img_sim(img_df, name3)

            # 3개 간 교집합
            img_score = list(set(img_score1) & set(img_score2) & set(img_score3))


            print('5단계', img_score[:10])
            # 이미지 유사도
            temp = sim_df[sim_df['name'].isin(img_score)]
            print('6단계', temp)
            print('7단계', temp.columns)
            temp = temp.sort_values(by='wv_cosine',ascending=False)
            temp = temp[:top_n]

            print('8단계', temp)

            temp = temp.sort_values(by=['wv_cosine','scaled_rating','year'], ascending=False)
            # print(temp.columns)
            print('9단계', temp)

            # print('7단계', type(temp['review_tagged_cleaned'][0]))

            classification = temp[['name', 'img', 'review', 'price']]
            print('10단계', classification)
            name = list(classification['name'])
            img = list(classification['img'])
            review = list(classification['review'])
            price = list(classification['price'])
            print('11단계', name, img)

            records = PredResults.objects.all()
            records.delete()

            for i in range(len(classification)):
                PredResults.objects.create(name=name[i], img=img[i], review=review[i] ,price=price[i])


            #### Wordcloud 만들기
            try : wordcloud(temp)
            except : pass

            end = timer()
            time = end - start

            return JsonResponse({'name': name, 'time' : time}, safe=False)

        except :
            return JsonResponse({'name': "해당되는 추천이 없습니다. 다시 입력해주세요"}, safe=False)

    else : return render(request, 'image_predict.html', {'list': category_list})



def view_results(request):
    # Submit prediction and show all

    data = PredResults.objects.all()

    return render(request, "results.html", {"dataset" : data})

def view_wordcloud(request):
    return render(request, "wordcloud.html")

def view_topic(request):
    return render(request, "topic_modeling.html")


# view 함수
# @csrf_exempt
# @api_view(['GET','POST'])
# def predict(request):
#
#     if request.POST.get('action') == 'post':
#
#         # Receive data from client(input)
#         # gender = str(request.POST.get('gender'))
#         # age = int(request.POST.get('age'))
#         main_category = str(request.POST.get('main_category'))
#         coordi = str(request.POST.get('coordi'))
#         input_text = str(request.POST.get('input_text'))
#         top_n = int(request.POST.get('topn'))
#
#         # 가방,모자,상의 <= 이런 양식으로 받아온다.
#         print(main_category)
#         print(coordi)
#
#         # coordi, category list 화
#         main_category = main_category.split(",")
#         coordi = coordi.split(",")
#
#         print('카테고리 :', main_category)
#         print('코디 :', coordi)
#         print('인풋 텍스트 :', input_text)
#         print('topn : ', top_n)
#
#         # Make prediction
#         try:
#             tot_result = recsys(main_category, coordi, input_text)
#
#             result = tot_result[:top_n]
#             result = result.sort_values(by=["wv_cosine", "scaled_rating", "year"], ascending=False)
#
#             print('1단계 :', result)
#             print('2단계 :', result.columns)
#
#             classification = result[['name', 'img', 'review', 'price']]
#             name = list(classification['name'])
#             img = list(classification['img'])
#             review = list(classification['review'])
#             price = list(classification['price'])
#             print(name)
#
#             records = PredResults.objects.all()
#             records.delete()
#
#             for i in range(len(classification)):
#                 PredResults.objects.create(name=name[i], img=img[i], review=review[i], price=price[i])
#
#             print('DB 저장 완료')
#
#             try:
#                 wordcloud(result)
#             except:
#                 pass
#
#             return JsonResponse({'name': name, 'img': img}, safe=False)
#
#         except:
#             return JsonResponse({'name': "해당되는 추천이 없습니다. 다시 입력해주세요"}, safe=False)
#
#     else:
#         return render(request, 'predict.html', {'category_list': category_list, 'coordi_list': coordi_list})


# @api_view(['GET'])
# def view_topic(request):
#     data = {'category_list': category_list, 'coordi_list': coordi_list}


# def view_results(request):
#     # Submit prediction and show all
#     data = PredResults.objects.all()
#     return render(request, "results.html", {"dataset" : data})

# custom을 생성하는 함수를 만들어야한다.