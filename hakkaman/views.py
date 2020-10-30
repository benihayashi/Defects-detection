from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse,JsonResponse
from .ml_model import predict_custom_img
import os
# Create your views here.
def home_view(request):
    context = {}
    if(request.method == "POST" and request.FILES["image"].name != "") :
        image = request.FILES["image"]
        extension = image.name.split(".")[1]
        if(extension != "jpg" and extension != "jpeg") :
            context["res"] = "Invalid File Type! Only jpg Files are accepted"
        else :
            fs = FileSystemStorage() 
            name = image.name.split(".")[0] + "." + "jpg"
            file = fs.save(name, image)
            prediction = predict_custom_img("./media/" + name)
            res = str(prediction)
            os.remove("./media/" + name)
            context["res"] = res
        return JsonResponse(context, status=200)

    return render(request,"home.html",context)