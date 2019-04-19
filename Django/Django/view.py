# -*- coding: utf-8 -*-

from django.shortcuts import render
import os
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from . import GetRectangle
import json

def select_pic(request):
    return render(request, "select.html")

@csrf_exempt
def upload_ajax(request):
     print(1111111)
     if request.method == 'POST':
            print(22222)
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
            result = str
            return HttpResponse(str)
