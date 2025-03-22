import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Load the CNN model
model = load_model("cnn_model_v2.keras")  # Load your CNN model

# Initialize session state counters if not already initialized
if "correct_count" not in st.session_state:
    st.session_state.correct_count = 0
if "incorrect_count" not in st.session_state:
    st.session_state.incorrect_count = 0

st.title("Digit Recognition using CNN")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="black",  # Set the background color to black
    stroke_color="white",  # Set the brush color to white
    stroke_width=20,  # Set the brush size
    background_color="black",  # Background color of the canvas
    height=280,  # Height of the canvas
    width=280,  # Width of the canvas
    drawing_mode="freedraw",  # Mode for free drawing
    key="canvas"
)

submit_button = st.button("Submit Drawing")

# Function to center the digit in a 20x20 box within a 28x28 image
def center_digit(image):
    image_array = np.array(image)
    
    # Find the bounding box of the digit
    rows = np.any(image_array < 255, axis=1)  # Look for non-white pixels
    cols = np.any(image_array < 255, axis=0)
    
    if not rows.any() or not cols.any():  # If no digit is drawn
        return image
    
    # Get the bounding box coordinates
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop the image to the bounding box
    cropped_image = image_array[rmin:rmax + 1, cmin:cmax + 1]

    # Resize the cropped image to 20x20
    resized_image = Image.fromarray(cropped_image).resize((20, 20))

    # Create a new 28x28 image with a black background
    centered_image = Image.new("L", (28, 28), 0)  # White background
    centered_image.paste(resized_image, (4, 4))  # Paste the digit in the center

    return centered_image

if submit_button and canvas_result.image_data is not None:
    # Convert to PIL Image, get only the canvas area
    image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
    
    # Convert to grayscale and remove alpha channel
    image = image.convert('L')  # 'L' is mode for grayscale
    
    # Center the digit in a 20x20 box within a 28x28 image
    centered_image = center_digit(image)

    # Display processed image
    st.image(centered_image, caption="Processed Image (Input to Model)", width=150, clamp=True, channels="L")

    # Convert the centered image to a numpy array
    image_array = np.array(centered_image)

    # Reshape to match model input shape (batch_size, height, width, channels)
    image_array = image_array.reshape(1, 28, 28, 1)

    # Make predictions
    predictions = model.predict(image_array)
    predicted_digit = np.argmax(predictions)

    # Show predicted digit
    st.write(f"### Predicted Digit: {predicted_digit}")

# Independent feedback widget
st.write("### Feedback")
feedback = st.radio("Was the last prediction correct?", ("No", "Yes"), key="feedback_radio")

# Add a persistent "Submit Feedback" button
if st.button("Submit Feedback"):
    if feedback == "Yes":
        st.session_state.correct_count += 1
    elif feedback == "No":
        st.session_state.incorrect_count += 1

# Show the current count of correct and incorrect predictions
st.write(f"### Correct Predictions: {st.session_state.correct_count}")
st.write(f"### Incorrect Predictions: {st.session_state.incorrect_count}")
