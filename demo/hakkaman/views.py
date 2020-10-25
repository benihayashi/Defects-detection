from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .ml_model import predict_custom_img
# Create your views here.
def home_view(request):
    context = {}
    if(request.method == "POST" and request.FILES["image"].name != "") :
        image = request.FILES["image"]
        extension = image.name.split(".")[1]
        if(extension != "png" and extension != "jpg" and extension != "jpeg") :
            context["res"] = "Invalid File Type!"
        else :
            fs = FileSystemStorage() 
            file = fs.save(image.name, image)
            prediction = predict_custom_img("./media/" + image.name)
            context["res"] = str(prediction)

    return render(request,"home.html",context)