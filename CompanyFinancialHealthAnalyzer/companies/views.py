from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
# Create your views here.

def companies(request):
    template = loader.get_template('myfirst.html')
    return HttpResponse(template.render())

def secondHTML(request):
    template = loader.get_template('mysecond.html')
    return HttpResponse(template.render())