# 🧠 Image Caption Generator using CNN + LSTM

This project implements an Image Caption Generator using deep learning techniques. It uses a Convolutional Neural Network (CNN) for extracting features from images and a Long Short-Term Memory (LSTM) network to generate relevant and meaningful captions.

## 📂 Dataset
The model is trained on the [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k), which contains:
- 8000 images
- 5 human-written captions per image

## 🧰 Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Pillow (PIL)
- Matplotlib
- Pre-trained Xception model (as CNN)

## 🚀 How It Works

1. **Data Preprocessing**
   - Load captions, clean text (lowercase, punctuation removal)
   - Add <start> and <end> tags to each caption

2. **Feature Extraction**
   - Use pre-trained Xception CNN to extract 2048-dimensional feature vectors

3. **Caption Encoding**
   - Tokenize captions and convert to integer sequences

4. **Model Architecture**
   - CNN features → Dense layer
   - Captions → Embedding → LSTM
   - Merge both → Dense layer → Output word (via softmax)

5. **Training**
   - Model is trained to predict the next word in a caption given the image and previous words

6. **Inference (Testing)**
   - A new image is passed to the model to generate a complete caption one word at a time

## 🧪 Example Output

Image: ![Sample](https://raw.githubusercontent.com/yourusername/yourrepo/main/sample.jpg)

**Generated Caption:**  
`"a man is riding a surfboard on a wave"`

## 📁 Folder Structure
```
📦 image_captioning_project/
 ┣ 📂 models/                 # Contains trained .h5 model
 ┣ 📄 tokenizer.p             # Tokenizer pickle file
 ┣ 📄 features.p              # Extracted image features
 ┣ 📄 descriptions.txt        # Cleaned caption data
 ┣ 📄 main.py                 # Code for training the model
 ┣ 📄 test.py                 # Code for generating captions from images
 ┗ 📄 README.md               # Project info
```

## ✅ Requirements
Install dependencies using pip:

```bash
pip install tensorflow keras numpy matplotlib pillow tqdm
```

## 📸 To Generate Captions
Place an image in the project folder and run:
```bash
python test.py --image test.jpg
```

## 📌 Future Improvements
- Add attention mechanism
- Use larger datasets (MS-COCO)
- Upgrade to transformers (e.g., BLIP)
- Deploy as a web/mobile application

## 👨‍💻 Author
Priyanshu

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
