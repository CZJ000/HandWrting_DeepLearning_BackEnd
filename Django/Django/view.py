# -*- coding: utf-8 -*-

from django.shortcuts import render
import os
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from . import GetRectangle

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
            status = 0
            result = str
            return HttpResponse(str)
def select_pic(request):
    return render(request, "select.html")

