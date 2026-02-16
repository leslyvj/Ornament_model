# Jewelry Damage Detector

A Streamlit application for detecting and analyzing damage in jewelry using AI.

## Features
- **Object Detection**: Detects jewelry types (Bangle, Necklace, Ring, etc.) using RT-DETR and YOLOv11 models.
- **AI Analysis**: Uses Gemini 2.0 Flash Lite (via OpenRouter) to inspect items for defects like missing gems or deformation.
- **Reporting**: Generates a detailed table of findings and allows JSON export.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/leslyvj/Ornament_model.git
   cd Ornament_model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory and add your OpenRouter API key:
   ```env
   OPENROUTER_API_KEY=your_api_key_here
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Models
- **RT-DETR**: Located in `models/rtdetr_best.pt`. Optimized for accuracy.
- **YOLOv11**: Located in `models/yolo_best.pt`. Optimized for speed.
