# Handwritten Digits Predictor using Dense Neural Network

This project is a web app for recognizing handwritten digits (0–9) using a dense neural network built with TensorFlow/Keras. Users can draw digits on a canvas and get instant predictions.

## Features
- Streamlit web interface with a drawable canvas
- Dense neural network trained on MNIST dataset
- Real-time digit prediction and probability visualization

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Handwritten_digits
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the model (optional, if you want to retrain):**
   ```bash
   python model/model.py
   ```
   This will save the trained model as `model/model.keras`.

2. **Run the Streamlit app:**
   ```bash
   streamlit run app/main.py
   ```

3. **Draw a digit on the canvas and click Predict.**

## Sample Demo

<video width="480" controls>
  <source src="./Screen Recording 2026-02-22 at 15.56.31.mov" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Project Structure
```
Handwritten_digits/
├── app/
│   └── main.py         # Streamlit app
├── model/
│   └── model.py        # Model definition and training
│   └── model.keras     # Saved Keras model
├── images/             # (Optional) Sample images
├── requirements.txt    # Python dependencies
└── Readme.md           # Project documentation
```

## Requirements
See `requirements.txt` for all dependencies. Main packages:
- streamlit
- streamlit-drawable-canvas
- tensorflow
- numpy
- pillow

## Model Details
- Dense neural network (fully connected layers)
- Trained on MNIST dataset
- Input: 28x28 grayscale image
- Output: Probability for each digit (0–9)

## Model Limitations
- Trained on MNIST dataset, numbers are centered and very thin brush strokes
- Fails to predict with thicker strokes
- Location dependent, better predictions around the center
- performs better with numbers like 8,4,2
- Next implementation in v2, using Convulational Neural Networks

## License
MIT License

## Credits
- TensorFlow/Keras
- Streamlit
- MNIST dataset
