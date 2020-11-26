# Chinese-stroke-order

---

The model is used to predict the stroke order of the chinese character.

---

## Usage : 

python glpredict.py image_path,

ex. python glpredict.py ./test_img/train/04eca.png

---

## Description : 

1. In the "test_img" directory, subdirectory "train" is training data, and "test" is testing data.

2. After you execute the program, 
the result will be put in the "output_Img" and the "output_Stk" directory.

3. The "output_Img directory stores the images of the stroke order of the chinese character.
   
   The "output_Stk" directory stores the x,y coordinates of the stroke order of the chinese character.
