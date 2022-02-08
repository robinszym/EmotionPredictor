# EmotionPredictor

The EmotionPredictor predicts the emotion distribution of images. It uses CLIP as backbone and a single linear unit model trained on the ArtEmis dataset.

# Install

```
git clone https://github.com/robinszym/EmotionPredictor.git
cd EmotionPredictor
pip install -e .
```
# Image affect prediction
The model predicts the emotion distribution of images.
![example](https://github.com/robinszym/EmotionPredictor/blob/beta/example.jpeg?raw=true)

## Web image demo

<p align = "center">
    <img src = "https://github.com/robinszym/EmotionPredictor/blob/main/image_prediction.png?raw=true">
</p>
<p align = "center">
Interface for images.
</p>

To create the demo simply run 
```
python emotionpredictor/demo_images.py
```
# Affective search
It is possible to perform affective search on images by imputing a text prompt and filtering with an emotion.
<p align = "center">
<img src = "https://github.com/robinszym/EmotionPredictor/blob/main/landscape_happy.jpeg?raw=true">
</p>
<p align = "center">
Result of the search "landscape" with each positive emotion.
</p>
## Demo
The demo is on its way. The embeddings have to be computed.


# Text affect
Since the model was trained with CLIP it is also possible to predict the affect of a word or a sentence. It's use for sentiment analysis is limited but is helpful for testing hypothesis on why the model predicts certain emotions. 

## Demo

```
python emotionpredictor/demo_text.py
```
Then click the local link generated. It takes a few seconds to load the clip model.

# About
This project was carried as part of my Master thesis at the IVRL lab of EPFL.



