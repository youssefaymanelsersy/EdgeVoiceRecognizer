# EdgeVoiceRecognizer: Isolated-Word Speech Recognition for ATmega32A

An end-to-end Machine Learning pipeline designed for resource-constrained 8-bit microcontrollers (specifically the AVR ATmega32A). This project handles data collection, feature extraction (MFCCs), model training (Decision Tree), and automatic C-code generation for embedded deployment.

## Features
- **Interactive CLI Menu:** A single unified interface (`main.py`) for the entire ML lifecycle.
- **Hardware Capture Modes:**
  - *Laptop Only:* Uses your computer's built-in microphone for rapid prototyping.
  - *Hybrid Mode:* Captures audio via the ATmega32A's ADC and streams it over Serial to the laptop for processing.
- **Optimized Feature Extraction:** Computes Mel-Frequency Cepstral Coefficients (MFCCs) optimized for fixed-point/low-memory C implementations.
- **Hyperparameter Autotuning:** Automatically sweeps across depths and splits to find the best-performing Decision Tree configuration.
- **Zero-Dependency C Export:** Automatically generates `model.h` containing pure C `if/else` logic that runs instantly on the ATmega32A without external libraries.

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**
   ```bash
   python main.py
   ```

3. **Workflow**
   - **Option 1:** Record your training dataset (20 samples per command).
   - **Option 10:** Record a separate test dataset (10 samples per command).
   - **Option 4:** Run the Autotuner to find the best model configuration.
   - **Option 3:** Test the model in real-time using your microphone.
   - **Option 8:** Export the trained model to C (`include/model.h`).

## Hardware Modes
By default, the script uses your laptop's microphone. To switch to the ATmega32A Serial microphone (Hybrid Mode), open `config.py` and change:
```python
CAPTURE_MODE = "serial"
SERIAL_PORT = "COM7" # Change to your MCU's port
```

## Advanced Settings
The `core/model.py` file contains an optional, highly tuned **Distance Rejection Algorithm** (currently commented out). When enabled, it calculates the statistical variance of your voice during training. If a random, out-of-vocabulary word (like "banana") is spoken, the algorithm mathematically proves it is an outlier and rejects it as `unknown`, preventing the Decision Tree from forcing a false positive.

## Repository Structure
- `main.py` - Primary CLI entry point
- `config.py` - Hardware and pipeline configuration
- `core/` - Audio processing, DSP, and ML logic
- `tools/` - Debugging and replay utilities
- `include/` - Generated C headers for Atmel Studio / CodeVision AVR
- `models/` - Serialized `.joblib` model artifacts
- `data/` - Raw `.json` audio recordings (Ignored by git)
