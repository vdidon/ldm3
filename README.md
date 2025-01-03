Info
====
`ldm.py 2025-01-01`

Python 3 version of (https://pypi.org/project/ldm)

`Author: vdidon <vdidon@live.fr>`

`Copyright: This module has been placed in the public domain.`

`version:0.1.0`
- `add compare2dir` 
`version:0.1.1`
- `add face_rec_ms` 
`version:0.1.3`
- `add compare2imglist`
`version:0.1.4`
- `support python3`

Classes:
- `LDM`: you can use a function to get landmarks and face feature  with no other libs 

Functions:

- `face_feature`: the feature of face in the image for face recognition 
- `landmarks`: get the landmarks and face in the image 
- `face_rec`: return the face similarity of the  face in two images
- `has_same_person`: return a int(>0) if there have a same person in two images,otherwise 0
- `compare2dir`: compare the face in the images of the two dirs,and return the rate ,score_avg,class_id1,class_id2
- `compare2imglist`: compare the face in the imglist ,and return the rate ,score_avg,class_id1,class_id2

How To Use This Module
======================
.. image:: funny.gif
   :height: 100px
   :width: 100px
   :alt: funny cat picture
   :align: center

1. example code:


.. code:: python

    imagepath="closed_eye/9.jfif"
    img=io.imread(imagepath)
    img1=img
    imagepath="closed_eye/14.jfif"
    img2=io.imread(imagepath)
    
    ldmer=ldm.LDM()
    print(img.shape[0])
    ldl,facel,txt=ldmer.landmarks(img)
    print(txt)
    
    for ld in ldl:
        print(10*'-')
        print('nose:')
        print(ld['nose'])
    for face in facel:
        print(10*'-') 
        print('face:')
        print(face.top())
        print(face.left())
        print(face.width())
        print(face.height())
        print(face.bottom())
        print(face.right())
        x,y,w,h=[face.top(),face.left(),face.width(),face.height()]
        print(x,y,w,h)
    print("feature:")
    ffl=ldmer.face_feature(img,facel)
    for ff in ffl:
        print(help(ff))
        print('ff='+str(ff))
        print('len(ff)='+str(len(ff)))
        print('ff[0]='+str(ff[0]))
        print('ff[127]='+str(ff[127]))

    print("face compare:")
    print(ldmer.face_rec(img1,img2))
    print(ldmer.has_same_person(img1,img2))
    print(ldmer.has_same_person(img2,img2))

    print("face number:")
    print(ldmer.face_number(img,facel))



Refresh
========

add a function : ldmer.imread(imgpath) 
modify  the return value number to: has_flag,max_score=ldmer.has_same_person(img1,img2)
