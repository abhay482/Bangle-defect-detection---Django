from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.utils import timezone
# Create your views here.

@login_required(login_url='account/signup')
def detect(request):
    if request.method == 'POST':
        import pickle
        import cv2
        import numpy as np
        from PIL import Image
        x = []
        with open('static/xgrf.pck', 'rb') as f:
                svm=pickle.load(f)

        img = _grab_image(stream=request.FILES["image"])
        rgb_planes = cv2.split(img)
        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((15,15), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)
        result = cv2.merge(result_planes) # normalize image
        result_norm = Image.fromarray( cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) #opencv to Image
        img_file = result_norm.resize((180,180), Image.ANTIALIAS) #resize image
        img_grey = img_file.convert('L')
        value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
        value = value.flatten()
        x.append(value)      
        ans= svm.predict(x)
        defect = ""
        if(ans==2):
            defect = "Perfect Bangle"

        if(ans==1):
            defect = "Broken Bangle"

        if(ans==0):
            defect = "Other defect"

        return render(request, 'defect/detect.html', {'ans': defect})

    else:
 
        return render(request, 'defect/detect.html', {'error': "Please Add a Image!!"})


def _grab_image(path=None, stream=None, url=None):
        import cv2
        import numpy as np
        # if the path is not None, then load the image from disk
        if path is not None:
            image = cv2.imread(path)
        # otherwise, the image does not reside on disk
        else:	
            # if the URL is not None, then download the image
            if url is not None:
                resp = urllib.urlopen(url)
                data = resp.read()
            # if the stream is not None, then the image has been uploaded
            elif stream is not None:
                data = stream.read()
            # convert the image to a NumPy array and then read it into
            # OpenCV format
            image = np.asarray(bytearray(data), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
        # return the image
        return image
