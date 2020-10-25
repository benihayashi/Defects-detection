from django.shortcuts import render

# Create your views here.
def home_view(request):
    context = {}
    if(request.method == "GET") :
        image = request.GET.get("image")
        context["res"] = image

    return render(request,"home.html",context)