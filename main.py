import os
import string
import numpy as np
from PIL import Image
from pickle import dump, load
from tqdm import tqdm
import tensorflow as tf
from keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.models import Model

# Adjust paths as per your system
dataset_text = r"C:\HUB\Image-captioning\Flickr8k_text1"
dataset_images = r"C:\HUB\Image-captioning\Flickr8k_Dataset"

def load_doc(filename):
    with open(filename, 'r') as file:
        return file.read()

def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.strip().split('\n')
    descriptions = {}
    for caption in captions:
        img, cap = caption.split('\t')
        img_id = img[:-2]
        if img_id not in descriptions:
            descriptions[img_id] = []
        descriptions[img_id].append(cap)
    return descriptions

def cleaning_text(captions):
    table = str.maketrans('', '', string.punctuation)
    for caps in captions.values():
        for i in range(len(caps)):
            desc = caps[i].replace("-", " ")
            desc = desc.split()
            desc = [w.lower().translate(table) for w in desc if w.isalpha() and len(w) > 1]
            caps[i] = ' '.join(desc)
    return captions

def save_descriptions(descriptions, filename):
    lines = []
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(f"{key}\t{desc}")
    with open(filename, "w") as file:
        file.write("\n".join(lines))

model_cnn = Xception(include_top=False, pooling='avg', input_shape=(299,299,3))

def extract_features(directory):
    features = {}
    valid_exts = {'.jpg', '.jpeg', '.png'}
    for img_name in tqdm(os.listdir(directory), desc="Extracting Features"):
        ext = os.path.splitext(img_name)[1].lower()
        if ext not in valid_exts:
            continue
        try:
            img_path = os.path.join(directory, img_name)
            img = Image.open(img_path).convert('RGB').resize((299,299))
            img = np.expand_dims(np.array(img), axis=0)
            img = preprocess_input(img)
            feat = model_cnn.predict(img, verbose=0)
            features[img_name] = feat
        except Exception as e:
            print(f"Skipping {img_name}: {e}")
    return features

def load_photos(filename):
    file = load_doc(filename)
    photos = file.strip().split('\n')
    return [p for p in photos if os.path.exists(os.path.join(dataset_images, p))]

def load_clean_descriptions(filename, photos):
    file = load_doc(filename)
    descriptions = {}
    for line in file.strip().split('\n'):
        words = line.split()
        if not words:
            continue
        image_id, image_caption = words[0], words[1:]
        if image_id in photos:
            if image_id not in descriptions:
                descriptions[image_id] = []
            descriptions[image_id].append('<start> ' + ' '.join(image_caption) + ' <end>')
    return descriptions

def load_features(photos):
    # First check if features.p exists
    if not os.path.exists("features.p"):
        print("Error: features.p does not exist. Please run the main function first.")
        return {}
    
    # Then load it
    all_features = load(open("features.p", "rb"))
    return {k: all_features[k] for k in photos if k in all_features}

def dict_to_list(descriptions):
    all_desc = []
    for desc_list in descriptions.values():
        all_desc.extend(desc_list)
    return all_desc

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)

def create_sequences(tokenizer, max_length, desc_list, feature, vocab_size):
    X1, X2, y = [], [], []
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def data_generator(descriptions, features, tokenizer, max_length, vocab_size, batch_size=32):
    while True:
        keys = list(descriptions.keys())
        np.random.shuffle(keys)
        for key in keys:
            desc_list = descriptions[key]
            feature = features.get(key)
            if feature is None:
                continue
            X1, X2, y = create_sequences(tokenizer, max_length, desc_list, feature[0], vocab_size)
            for i in range(0, len(X1), batch_size):
                yield ([X1[i:i+batch_size], X2[i:i+batch_size]], y[i:i+batch_size])

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model

def steps_per_epoch(descriptions, batch_size=32):
    total_seqs = 0
    for descs in descriptions.values():
        for desc in descs:
            total_seqs += len(desc.split()) - 1
    return total_seqs // batch_size

# Remove the tqdm().pandas() line that's causing the warning
# The code works fine without it

# CHECK THIS LINE: Don't try to load features.p directly at the global level
# We need to ensure all features.p access happens inside functions or the main block

def main():
    captions_file = os.path.join(dataset_text, "Flickr8k.token.txt")
    print("Loading captions...")
    descriptions = all_img_captions(captions_file)
    descriptions = cleaning_text(descriptions)
    save_descriptions(descriptions, "descriptions.txt")
    print(f"Total descriptions: {len(descriptions)}")

    # Check if features.p exists before trying to load it
    if os.path.exists("features.p"):
        print("Loading features from features.p file...")
        features = load(open("features.p", "rb"))
    else:
        print("Extracting features from images (this may take a while)...")
        features = extract_features(dataset_images)
        print("Saving extracted features to features.p...")
        dump(features, open("features.p", "wb"))
    print(f"Features extracted/loaded: {len(features)}")

    train_images_file = os.path.join(dataset_text, "Flickr_8k.trainImages.txt")
    train_imgs = load_photos(train_images_file)
    train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
    train_features = load_features(train_imgs)
    print(f"Train images: {len(train_imgs)}")
    print(f"Train descriptions: {len(train_descriptions)}")
    print(f"Train features: {len(train_features)}")

    tokenizer = create_tokenizer(train_descriptions)
    dump(tokenizer, open('tokenizer.p', 'wb'))
    vocab_size = len(tokenizer.word_index) + 1
    max_len = max_length(train_descriptions)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Max caption length: {max_len}")

    model = define_model(vocab_size, max_len)

    batch_size = 32
    steps = steps_per_epoch(train_descriptions, batch_size)

    generator = data_generator(train_descriptions, train_features, tokenizer, max_len, vocab_size, batch_size)

    if not os.path.exists("models2"):
        os.mkdir("models2")

    epochs = 10
    for i in range(epochs):
        print(f"Epoch {i+1}/{epochs}")
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        model.save(f"models2/model_{i}.h5")

# THIS IS CRITICAL: Make sure to call the main function
if __name__ == "__main__":
    main()