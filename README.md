# Bengaluru Air Quality Map & Pattern Explorer (Tkinter)

**Team Infinite Loop**  
- Nilasha B (`1BM25IM031`)
- Abdulhai Nalband (`1WA25CS005`)
- Adamya Jayaprakash Kulkarni (`1BM25CS323`)
- Aditya Sanjeev Lokare (`1WA25CS019`)

## What It Does
A clean, 3-tab Tkinter + Matplotlib desktop app aligned to our presentation:
1. **Interactive Bengaluru Map**: View real CPCB/KSPCB AQI across 14 Bengaluru neighborhoods (`Peenya`, `Silk Board`, `City Railway Station`, `Bapuji Nagar`, `Jayanagar 5th Block`, `BTM Layout`, `Hebbal`, `Kadabesanahalli`, `RVCE-Mailasandra`, `Hombegowda Nagar`, `Sanegurava Halli`, `Shivapura Peenya`, `Kasturi Nagar`, `Jigani`). Filter by **Month (`Jan–Dec`)**, **Season**, or click **`Play Year (Jan-Dec)`** to animate seasonal changes; click any pin on the map to see its worst months and local pollution causes.
2. **Worst Times, Heavy Metals & Causes**: Visualizes monthly curves and real CPCB Diwali festival spikes alongside explanations of firework-borne heavy metals (`Barium`, `Strontium`, `Potassium`, `Copper`, `Aluminium/Magnesium`, `Lead`), winter thermal inversions, leaf/waste burning, and area-wise fuel burning.
3. **Data Cleaning & Simple Predictor**: Demonstrates Pandas raw-to-clean data preparation (handling mixed timestamps, duplicates, and sensor glitches) and a Regression model (`Linear Regression` / `Random Forest`) with MAE, RMSE, R², and an interactive Area + Month AQI estimator.

## Run
```cmd
.\.venv\Scripts\python.exe main.py
```

## Run Tests
```cmd
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
