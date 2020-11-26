# Chinese-stroke-order

---

The model is used to predict the stroke order of the chinese character.

---

## Usage : 

python glpredict.py image_path,

ex. python glpredict.py ./test_img/train/04eca.png

---

## Description : 

In the "test_img" directory, subdirectory "train" is training data, and "test" is testing data.

After you execute the program, 
the result will be put in the "output_Img" and the "output_Stk".

The "output_Img directory stores the images of the stroke order of the chinese character,
and the "output_Stk" directory stores the x,y coordinates of the stroke order of the chinese character.
