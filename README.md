# <div align="center"> CS B551 - Assignment 4: Machine Learning
####  <div align="center"> CSCI B551 - Elements of Artificial Intelligence

<br>

###### Name: Sreesha Srinivasan Kuruvadi
###### Email: *sskuruva@iu.edu*
<br>

***
### [Part 1: K-Nearest Neighbors Classification](https://github.iu.edu/cs-b551-fa2021/sskuruva-a4/tree/master/k_nearest_neighbors.py)
***

#### Problem Statement:
Your goal in this part is to implement a k-nearest neighbors classifier from scratch. Your GitHub repository contains the skeleton code for two files that will be used to implement the algorithm: utils.py and k_nearest_neighbors.py.


***
### [Part 2: Multilayer Perceptron Classification](https://github.iu.edu/cs-b551-fa2021/sskuruva-a4/tree/master/multilayer_perceptron.py)

***

#### Problem Statement:
Your goal in this part is to implement a feedforward fully-connected multilayer perceptron classifier with one hidden layer (as shown in the description above) from scratch. As before, your GitHub repository contains the skeleton code for two files that will be used to implement the algorithm: utils.py and multilayer_perceptron.py.

The multilayer_perceptron.py file defines the MultilayerPerceptron class that we will use to implement the algorithm from scratch. Just like the previous part, the __init__ function has already been properly implemented for you. The attributes for the class itself are described in detail in the skeleton code. When creating the MultilayerPerceptron object, the following parameters must be specified (or their default values will be used):
• n_hidden: the number of neurons in the one hidden layer of the neural network.
• hidden_activation: represents the activation function of the hidden layer (can be either ‘identity’,
‘sigmoid’, ‘tanh’, or ‘relu’).
• n_iterations: represents the number of gradient descent iterations performed by the fit(X, y)
method.
• learning_rate: represents the learning rate used when updating neural network weights during gra- dient descent.
#### Command:
<code>  python3 ./polar.py input_file.jpg airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord
 </code>

#### Approach
##### 1. In the simple model, for air-ice we used numpy library to find the maximum row index in every column using the numpy.argmax() function.
And for ice-rock boundary -
- Set air ice boundary edge strength to zero.
- numpy library to find the maximum row index in every column using the numpy.argmax() function.
##### 2. For the HMM-viterbi model, to populate the transition probabilities, we used viterbi algorithm with dynamic programming and back tracking method.

##### 3. When additional human feedback is given we just tweaked the previous logic, made that particular pixel's state probability maximum, so the resultant path will be the path that passes through the human labeled pixel always. 
- set the p_distribution corresponding to the pixel to 1
- recur right
- recur left
- update max probability coordinates through backward algorithm
- repeat the same for ice-rock boundary

##### 4. Transition offset probabilities are used to decrease the probability of pixels far away from the current row
- Picked the below two values
  - trans_1 = [0.5, 0.4, 0.1] - causes aggressive decrease in probabilities
  - trans_2 = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.0005, 0.00005, 0] - close to a Guassian distribution.
  - trans_2 works better than trans1 as it considers more possibilities on the two sides of the row and less aggressive nearer to the row and exponentially aggressive farther away from the row.


###### Original Image
![Original Image](https://github.iu.edu/cs-b551-fa2021/amanurk-arushuk-sskuruva-a3/blob/master/part2/test_images/23.png)

<code>python3 ./polar.py test_images/23.png 147 35 148 68</code>

###### Using - [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.0005, 0.00005, 0]
![Air-ice boundary](https://github.iu.edu/cs-b551-fa2021/amanurk-arushuk-sskuruva-a3/blob/master/part2/air_ice_output.png)
![Ice-rock boundary](https://github.iu.edu/cs-b551-fa2021/amanurk-arushuk-sskuruva-a3/blob/master/part2/ice_rock_output.png)

##### Using - [0.5, 0.4, 0.1]
![Air-ice boundary](https://github.iu.edu/cs-b551-fa2021/amanurk-arushuk-sskuruva-a3/blob/master/part2/air_ice_output.png)
![Ice-rock boundary](https://github.iu.edu/cs-b551-fa2021/amanurk-arushuk-sskuruva-a3/blob/master/part2/ice_rock_output_low_transition.png)

#### References:
##### 1. http://www.cs.cmu.edu/~guestrin/Class/10701/slides/hmms-structurelearn.pdf
##### 4. https://web.stanford.edu/~jurafsky/slp3/A.pdf
##### 5. Canvas lecture slides

### [Part 3: Reading text](https://github.iu.edu/cs-b551-fa2021/amanurk-arushuk-sskuruva-a3/tree/master/part3)


#### Problem Statement
To show the versatility of HMMs, let’s try applying them to another problem; if you’re careful and you plan ahead, you can probably re-use much of your code from Parts 1 and 2 to solve this problem. 
Our goal is to recognize text in an image – e.g., to recognize that Figure 2 says “It is so ordered.” But the images are noisy, so any particular letter may be difficult to recognize. However, if we make the assumption that these images have English words and sentences, we can use statistical properties of the language to resolve ambiguities, just like in Part 2.

#### Approach
Naive Bayes Assumption based probabilities are the key to good Image Character Recognition. The emission probabilities without the Naive Bayes assumption works well only for clean images of characters. However, our code works significantly well even for the images with lot of noise in addition to clear character pixel values being present. This happens when we only match the grids pixels which match in the test and train images. Having defined the probabilities to the "unmatched" factor greatly improves recognition of images with sparse pixels (eg : test-16.png test-8.png).

We have also factored in the blanks or the " " pixels separately with a minimum probability factor as it acts like a string cleaning mechanism, considering the blank spaces in the image at the places which do not outline a character, can help us to detect a character in case of sparsely pixelated image. Tweaking the weight values given to match, unmatch, and blanks is the key to tradeoff between making the model better for detecting the noisy, or sparse, images.

The current values of weights is better at detecting sparsely pixelated images but can give some wrong predictions for the noisy images. But this can be varied by changing weights values on the line # 61 of char_recognizer :
''' prob = (0.70 * match_score + 0.25 * blanks + 0.05 * unwatch_score) / (CHARACTER_WIDTH * CHARACTER_HEIGHT) # 0.75 * match_prob + 0.20 * blanks + 0.05 * (1 - match_prob) is given to weigh the matches, misses and blanks in the grid comparisons. Basing this on Naive Bayes assumption of having some prior probabilities'''. These values can be tweaked for testing too
Viterbi algorithm really helps in improving the probabilities of proper words. For eg: UN1TED predicted by the HMM gets converted to UNITED when run with Viterbi.
We are able to predict almost humanly unreadable images, with considerable accuracy. However, some images (eg. test-15.png) do give unwanted prediction because of the transitional probability consideration. For completeness, the essence of emmision probability in this use case if the probability of matching of test image character with each training image characters (given in courier.png say). The transition probability is the probability of occurance of one letter after a specific letter calculated based on the text training set.
