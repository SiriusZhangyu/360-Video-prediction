360 degree video view prediction is one of the VR applications and is quite useful in daily lives to predict the future 
FOV(field of view) in a long time horizon rather than a few seconds. We will use heatmap-based model[1] to predict.  
The heatmaps model will use the equirectangular heatmaps to represent the FOV center distributions.
For using the target user information only, we used a seq2seq model where the encoder and decoder each uses a convLSTM. 
Then, FCN will be applied to concatenate hidden state maps of all the layers to generate the results of prediction. 
We are also inspired to improve the prediction by using other user’s views or applying a different heatmap shape. 
Furthermore, we will try to do the gaussian heatmaps on the sphere and project the heatmaps to the equirectangular images.


