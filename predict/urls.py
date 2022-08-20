from django.urls import path, re_path
from . import views

app_name = "predict"

urlpatterns = [
    # predict_page1 predict_page2
    path('', views.predict_page1, name='prediction_page'),
    path('img/', views.predict_page2, name='image_prediction_page'),

    path('predict/', views.predict, name='prediction'),
    path('img-predict/', views.img_predict, name='image_prediction'),
    # path('item-predict/', views.item_predict, name='item_prediction_page'),

    path('topics/', views.view_topic, name='topics'),

    # path('results/', views.view_results, name='results'),
    # drf result ~ detail view ex)728
    path('results/', views.ViewResult.as_view(), name='results'),
    path('results/<int:pk>', views.SeeItemView.as_view(), name='results_detail'),

    path('view_results/', views.PostListView.as_view(), name='view_results'),

    # path('view_results/', views.view_results, name='view_results'),
    path('view_results_sort/', views.view_sorted_results, name='sorted_view_results'),

    re_path(r'^product/', views.ViewProduct.as_view(), name='product'),

    path('wordcloud/', views.view_wordcloud, name='wordcloud'),
]


