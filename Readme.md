This is just the classifier from the NTM project.  
You should install it by going to the root and calling:
```
pip3 install -e .
```
Then use it to import classify image in your python runtime:  
```
from ntm_classifier.classify import classifiy_image  
classifiy_image(some_png_img)
```