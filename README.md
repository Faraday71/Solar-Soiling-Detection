# Solar Panel Fault Detection: Dust vs. Shading

## What is this?
This project uses machine learning to solve a real problem in solar:figuring out exactly why a solar panel is losing power. It analyzes standard inverter data to tell the difference between slow dust buildup and sudden physical shadows.

## The Problem
When a solar panel produces less power than expected, standard monitoring systems just see a drop in output. They can't always tell why it happened.
* **Dust (Soiling):** Builds up slowly over weeks, causing a gradual drop in current. It usually washes away when it rains.
* **Shading:** Happens when a cloud passes or a tree blocks the sun. This triggers the panel's internal bypass diodes, causing a sudden, sharp drop in voltage.

If a solar company can't tell the difference, they might waste money sending a cleaning crew out just because a cloud passed by. 

## How I Solved It
I built a deep learning model to look at the history of the panel's performance and recognize the physical signature of each problem. 

1. **The Data:** I used real, open-source solar telemetry data from the National Renewable Energy Laboratory (NREL PVDAQ System 10). 
2. **Physics-Based Simulation:** Because the original data was perfectly clean, I wrote a Python script to inject realistic faults into the data. I added a slow daily decay for dust and sharp morning voltage drops for shade.
3. **The Model:** I built a Long Short-Term Memory (LSTM) neural network using PyTorch. Instead of looking at one single moment in time, the model looks at a rolling 60-minute window of data (Voltage, Current, Irradiance, and Temperature) to understand the trend.

## Results
The LSTM model achieved **98.7% accuracy** on unseen test data. It successfully learned the underlying physics and can accurately classify the panel's state into four categories:
* `Normal`
* `Dusty`
* `Shaded`
* `Shaded_and_Dusty`

## Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Environment:** Jupyter Notebook, Google Colab (T4 GPU for training)

## Repository Structure
* `/data/processed/` - The cleaned and labeled CSV data used for training.
* `/models/` - The saved PyTorch model weights (`.pth`), so you don't have to retrain from scratch.
* `/notebooks/` - The step-by-step Jupyter notebooks (Data Exploration, Fault Injection, Training, and Inference).

## How to Run It
If you want to test the model on your own machine without waiting hours for it to train:
1. Clone this repository.
2. Install the required libraries (PyTorch, Pandas, Scikit-Learn).
3. Open `notebooks/04_model_inference.ipynb` and run the cells. It will load the pre-trained weights from the `/models/` folder and make a live prediction on a sample data sequence.