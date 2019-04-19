# -*- coding: utf-8 -*-

from django.shortcuts import render
from django.views.decorators import csrf
import os
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.csrf import csrf_exempt
import base64
from . import GetRectangle
import json

# 接收POST请求数据
def search_post(request):
    ctx = {}
    if request.POST:
        ctx['rlt'] = request.POST['q']
    return render(request, "post.html", ctx)


def select_pic(request):
    return render(request, "select.html")

@csrf_exempt
def upload_ajax(request):
        if request.method == 'POST':
            img = request.POST.get('image')
            #print(img)
            header = "data:image/png;base64,"
            img = img[len(header):]
            imagedata = base64.b64decode(img)
            #print(imagedata)
            path='./1.png'
            file = open(path, "wb")
            file.write(imagedata)
            file.close()
            str=GetRectangle.CutImgAndRecognize(path)
            print(str)
            # f = open(os.path.join('static', img.name), 'wb')
            # for chunk in img.chunks():
            #     f.write(chunk)
            # f.close()
            status = 0
            result = "Error!"
            return HttpResponse(str)
