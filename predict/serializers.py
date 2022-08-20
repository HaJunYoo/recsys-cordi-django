from rest_framework import serializers
from .models import *

class PredSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredResults
        fields = ['id', 'name', 'img', 'review', 'price',
                  'man', 'woman','rating','coordi', 'cosine_sim']


class PredDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredResults
        fields = ['id', 'name', 'img', 'review',
                  'price', 'man', 'woman','rating','coordi', 'cosine_sim']


class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ['id', 'name', 'img', 'review', 'price',
                  'man', 'woman','rating','coordi',
                  'cosine_sim', 'like',
                  'age18', 'age19_23', 'age24_28','age29_33', 'age34_39',	'age40',
                  'only_season' ]