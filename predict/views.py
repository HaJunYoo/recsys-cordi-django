from collections import OrderedDict

from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import render,  get_object_or_404
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import ListView

from predict.embedding import *
from predict.list import *
from predict.models import PredResults, Product, Wishlist, Custom
from predict.serializers import PredSerializer
from predict.serializers import PredDetailSerializer

### RestFramework
from rest_framework.decorators import api_view
from rest_framework.generics import GenericAPIView, RetrieveAPIView
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response
from rest_framework import generics, filters
# from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.parsers import JSONParser

## Third Party
import json
from nltk.tokenize import word_tokenize
# from konlpy.tag import Okt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import ast
from timeit import default_timer as timer



'''
데이터 프레임 -> 로직 -> 추천 -> 결과 

뷰(view)   
'''
#############################################################
# => 데이터 선언
#   코사인유사도를 구한 행렬을 역순으로 정렬 -> 유사도가 높은 순의 인덱스로 정렬됨
#   시간 복잡도가 제일 오래 걸림 => 여기서 시간을 제일 많이 소모됨 => O(n*logn)
#   sim_sorted_ind=sim.argsort()[:,::-1]
# sim_sorted_ind = np.load('predict/data/sim_sorted_after.npy', allow_pickle=True)
# sim_sorted_ind => embeddings.py
df = pd.read_csv('predict/data/merged_df_최종.csv', index_col='Unnamed: 0')
img_df = load_img_feature()

################################################################
# embeddings.py에서 함수들 임포트

###################################################

df["tags"] = df["tags"].apply(lambda x: make_list(x))
df["review_tagged_cleaned"] = df["review_tagged_cleaned"].apply(lambda x: make_list(x))
df["coordi"] = df["coordi"].apply(lambda x: make_list(x))

######## mean_vector.npy 불러오는 것으로 아래 코드 해결

# topic modeling
# embedding.py
#

## Views.py ####################################

def predict_page1(request):
    return render(request, 'predict.html',
           {'category_list': category_list, 'coordi_list': coordi_list})

def predict_page2(request):
    return render(request, 'image_predict.html', {'list': category_list})

# view 함수
@csrf_exempt
@api_view(['POST'])
def predict(request):
    if request.method == 'POST':
        # unparsed_json = request.body.decode('utf-8')
        # postData = json.loads(unparsed_json)
        # print(postData)
        data = JSONParser().parse(request)
        print(data)
        print(data['main_category'])

        # {'main_category': '상의', 'coordi': '캐주얼', 'input_text': '여름', 'top_n': '30'}

        main_category = str(data['main_category']) # 로직, 로직
        coordi = str(data['coordi']) # 코디,코디
        input_text = str(data['input_text']) # input_text
        top_n = int(data['top_n']) # 50

        # 가방,모자,상의 <= 이런 양식으로 받아온다.
        # print(main_category)
        # print(coordi)

        if input_text == '가성비' :
            input_text = '가성'

        # coordi, category list 화
        main_category = main_category.split(",")
        coordi = coordi.split(",")

        print('카테고리 :', main_category)
        print('코디 :', coordi)
        print('인풋 텍스트 :', input_text)
        print('topn : ', top_n)

        # Make prediction
        try:
            tot_result = recsys(main_category, coordi, input_text)

            result = tot_result[:top_n]
            result = result.sort_values(by=["wv_cosine", "scaled_rating", "year"], ascending=False)

            print('1단계 :', result)
            print('2단계 :', result.columns)

            classification = result[['name', 'img', 'review', 'price', 'man', 'woman', 'scaled_rating', 'coordi', 'wv_cosine']]
            classification["coordi"] = classification["coordi"].apply(lambda x: make_list(x))

            name = list(classification['name'])
            img = list(classification['img'])
            review = list(classification['review'])
            price = list(classification['price'])
            woman = list(classification['woman'])
            man = list(classification['man'])
            rating = list(classification['scaled_rating'])
            coordi = list(classification['coordi'])  # coordi는 리스트 of 리스트
            cosine_sim = list(classification['wv_cosine'])

            print(name)
            print(coordi)

            records = PredResults.objects.all()
            records.delete()

            for i in range(len(classification)):
                PredResults.objects.create(name=name[i], img=img[i],
                                           review=review[i], price=price[i],
                                           woman=woman[i], man=man[i],
                                           rating=rating[i], cosine_sim=cosine_sim[i],
                                           coordi=', '.join(coordi[i]),
                                           # coordi={'coordi': coordi[i]}
                )
                # coordi = {'coordi': coordi[i]})
            print('DB 저장 완료')

            try:
                wordcloud(result)
            except:
                pass

            return JsonResponse({'name': name}, safe=False)

        except:
            return JsonResponse({'name': "해당되는 추천이 없습니다. 다시 입력해주세요"}, safe=False)

    # else:
    #     return render(request, 'predict.html', {'category_list': category_list, 'coordi_list': coordi_list})


# image classification view 함수
@csrf_exempt
@api_view(['GET', 'POST'])
def img_predict(request):

    if request.method == 'POST':
        data = JSONParser().parse(request)
        print(data)
        input_text = str(data['input_text'])
        top_n = int(data['top_n'])
        main_category = str(data['main_category'])
        print(input_text)
        print(top_n)
        print(main_category)
        print()
        if input_text == '가성비' :
            input_text = '가성'

        # input_text = data['input_text']
        # top_n = data['top_n']
        # main_category = data['main_category']

        print(input_text, top_n, main_category)
        # Make prediction
        try:
            start = timer()
            # Okt가 시간이 엄청 걸림

            string_list = word_tokenize(input_text)  # 리스트 형태

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
            temp = temp.sort_values(by='wv_cosine', ascending=False)
            temp = temp[:top_n]

            print('8단계', temp)

            temp = temp.sort_values(by=['wv_cosine', 'scaled_rating', 'year'], ascending=False)
            # print(temp.columns)
            print('9단계', temp)

            # print('7단계', type(temp['review_tagged_cleaned'][0]))

            classification = temp[['name', 'img', 'review', 'price','man', 'woman', 'scaled_rating', 'coordi', 'wv_cosine']]

            name = list(classification['name'])
            img = list(classification['img'])
            review = list(classification['review'])
            price = list(classification['price'])
            woman = list(classification['woman'])
            man = list(classification['man'])
            rating = list(classification['scaled_rating'])
            coordi = list(classification['coordi']) # coordi는 리스트 of 리스트
            cosine_sim = list(classification['wv_cosine'])

            print(name)
            print(coordi)

            records = PredResults.objects.all()
            records.delete()

            for i in range(len(classification)):
                PredResults.objects.create(name=name[i], img=img[i],
                                           review=review[i], price=price[i],
                                           woman= woman[i], man = man[i],
                                           rating= rating[i], cosine_sim = cosine_sim[i],
                                           coordi= ', '.join(coordi[i]) )
                                           # coordi= {'coordi': coordi[i]} )
                                           # coordi = {'coordi': coordi[i]})
            #### Wordcloud 만들기
            try:
                wordcloud(temp)
            except:
                pass

            end = timer()
            time = end - start

            return JsonResponse({'name': name, 'time': time}, safe=False)

        except:
            return JsonResponse({'name': "해당되는 추천이 없습니다. 다시 입력해주세요"}, safe=False)


# @api_view(['GET', 'POST'])
# def view_results(request):
#     # Submit prediction and show all
#     if request.method == "GET":
#         data = PredResults.objects.all()
#         serializer = PredSerializer(data, many=True)
#         # many => queryset에 대응. many 없으면 instance 1개가 올 것으로 기대하고 있어 에러 발생함.
#         return Response(serializer.data)


# @api_view(['GET', 'POST'])
@csrf_exempt
def view_sorted_results(request):
    # data = JSONParser().parse(request)
    data = request.GET.get('sort')
    sorted_data = PredResults.objects.all()
    print(data)
    if data == 'rating' :

        sorted_data = sorted_data.order_by('-rating')

        print(sorted_data)
        return render(request, "results.html", {"dataset": sorted_data})

    elif data == 'man' :

        sorted_data = sorted_data.order_by('-man')

        print(sorted_data)
        return render(request, "results.html", {"dataset": sorted_data})

    elif data == 'woman' :

        sorted_data = sorted_data.order_by('-woman')

        print(sorted_data)
        return render(request, "results.html", {"dataset": sorted_data})

    else :
        sorted_data = sorted_data.order_by('-cosine_sim')
        return render(request, "results.html", {"dataset": sorted_data})


class PostListView(ListView):
    model = PredResults
    template_name = 'results.html'
    # FOLLOW THIS NAMING SCHEME <app>/<model>_<viewtype>.html
    context_object_name = 'dataset'
    ordering = ['-cosine_sim']

    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return super(PostListView, self).dispatch(request, *args, **kwargs)

    # def get_ordering(self):
    #     ordering = self.request.GET.get('rating_sort')
    #     print(ordering)
    #     if ordering is not None :
    #         #Order live feed events according to closest start date events at the top
    #         return ordering
    #     else :
    #         ordering = '-rating'
    #         return ordering

def view_results(request):
    # Submit prediction and show all
    data = PredResults.objects.all()
    return render(request, "results.html", {"dataset" : data})




class PostPageNumberPagination(PageNumberPagination):
    page_size = 8
    # page_size_query_param = 'page_size'
    # max_page_size = 1000

    def get_paginated_response(self, data):
        return Response(OrderedDict([
            ('postList', data),
            ('pageCnt', self.page.paginator.num_pages), # 수정
            ('CurPage', self.page.number), # 수정
        ]))



class ViewResult(generics.ListCreateAPIView):
    queryset = PredResults.objects.all()
    serializer_class = PredSerializer
    pagination_class = PostPageNumberPagination
    # filter_backends = [DjangoFilterBackend]
    # filterset_fields = ['woman', 'man']
    filter_backends = [filters.OrderingFilter, filters.SearchFilter]
    ordering_fields = ['woman', 'man', 'rating', 'cosine_sim']
    search_fields = ['coordi']

    def list(self, request, *args, **kwargs):
        # Note the use of `get_queryset()` instead of `self.queryset`
        queryset = self.filter_queryset(self.get_queryset())
        serializer = PredSerializer(queryset, many=True)
        return Response(serializer.data)


class SeeItemView(RetrieveAPIView):
    queryset = PredResults.objects.all()
    serializer_class = PredDetailSerializer


class ViewProduct(generics.ListCreateAPIView):
    queryset = Product.objects.all()
    serializer_class = PredSerializer

    pagination_class = PostPageNumberPagination

    def get_queryset(self):
        qs = super().get_queryset()
        product_name = self.request.query_params.get('name', None)
        if product_name :
            qs = qs.filter(name__icontains = product_name)

        return qs



# 추천된 항목에서 선택된 항목들을 wishlist에 추가하는 함수
@api_view(['GET', 'POST'])
def choice_results(request):
    if request.method == "POST":
        recommend_name = str(request.POST.get('recommend_name'))
        custom = str(request.POST.get('custom'))
        product = Product.objects.filter(Q(name__contains = recommend_name))[0]
        custom = Custom.objects.filter(Q(custom_name__contains = custom))[0]
        if recommend_name == product.name :
            item = get_object_or_404(Product, pk=product.id)
            Wishlist.objects.create(custom=custom, product = item)


def view_wordcloud(request):
    return render(request, "wordcloud.html")


def view_topic(request):
    return render(request, "topic_modeling.html")
