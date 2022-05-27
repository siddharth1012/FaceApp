from django.shortcuts import render, redirect
from django.http import HttpResponse
from app.forms import FaceRecognitionform
from app.machinelearning import pipeline_model
from django.conf import settings
from app.models import FaceRecognition
import os
from app.machinelearning import key_feature


#Gives result by calling pipeline_model 
def resultGiverFunc(request,form,save):
    # extract the image object from database
    primary_key = save.pk
    imgobj = FaceRecognition.objects.get(pk=primary_key)
    fileroot = str(imgobj.image)
    #for fetching the address of image
    filepath = os.path.join(settings.MEDIA_ROOT,fileroot)
    results = pipeline_model(filepath)
    #returns results with all the details regarding an image
    return results

#To render index upon request
def index(request):
    return render(request,'index.html')


#View that returns images and score of face detected
def detection(request):
    form = FaceRecognitionform()

    if request.method == 'POST':
        form = FaceRecognitionform(request.POST or None, request.FILES or None)
        #checks if the form is valid
        if form.is_valid():
            #if the form is valid then save the form in save
            save = form.save(commit=True)
            #calling resultGiverFunc to fetch results
            results = resultGiverFunc(request,form,save)
            print(results)
        return render(request, 'detect/detect.html',{'form':form,'upload':True,'results':results})
    return render(request,'detect/detect.html',{'form':form,'uplaod':False})

#View that returns images with name and face match ratio of face detected
def recognition(request):
    form = FaceRecognitionform()

    if request.method == 'POST':
        form = FaceRecognitionform(request.POST or None, request.FILES or None)
        if form.is_valid():
            save = form.save(commit=True)
            #calling resultGiverFunc to fetch results
            results = resultGiverFunc(request,form,save) 
            print(results)
        return render(request, 'recognize/recognize.html',{'form':form,'upload':True,'results':results})
    return render(request,'recognize/recognize.html',{'form':form,'uplaod':False})

#View that returns images with emotions of the person and ratio of face detected
#highest accuracy of the model is 34%
def emotions(request):
    form = FaceRecognitionform()

    if request.method == 'POST':
        form = FaceRecognitionform(request.POST or None, request.FILES or None)
        if form.is_valid():
            save = form.save(commit=True)
            #calling resultGiverFunc to fetch results
            results = resultGiverFunc(request,form,save)
            print(results)
        return render(request, 'emotions/emotions.html',{'form':form,'upload':True,'results':results})
    return render(request,'emotions/emotions.html',{'form':form,'uplaod':False})

def facialfeatures(request):
    form = FaceRecognitionform()

    if request.method == 'POST':
        form = FaceRecognitionform(request.POST or None, request.FILES or None)
        if form.is_valid():
            save = form.save(commit=True)
            # extract the image object from database
            primary_key = save.pk
            imgobj = FaceRecognition.objects.get(pk=primary_key)
            fileroot = str(imgobj.image)
            filepath = os.path.join(settings.MEDIA_ROOT,fileroot)
            #calling key_feature function from models to fetch results
            key_feature(filepath)
        return render(request, 'facialfeatures/facialfeatures.html',{'form':form,'upload':True})
    return render(request,'facialfeatures/facialfeatures.html',{'form':form,'uplaod':False})

#view for rendering the view page that has timeline
def view_timeline(request):
    return render(request, 'view/view.html')