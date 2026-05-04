from __future__ import annotations

from core.pipeline import SAMPLE_RATE, SAMPLE_LENGTH, FRAME_LENGTH, FRAME_COUNT

# Project-wide constants
VOICE_SAMPLE_RATE = SAMPLE_RATE
VOICE_SAMPLE_LENGTH = SAMPLE_LENGTH
VOICE_FRAME_LENGTH = FRAME_LENGTH
VOICE_FRAME_COUNT = FRAME_COUNT

# Dataset sizes
TRAINING_SAMPLES_PER_COMMAND = 20
TEST_SAMPLES_PER_COMMAND = 10

# Default MFCC configuration
DEFAULT_N_MFCC = 10

# Hardware Capture Mode
# Set to "microphone" to use your Laptop's internal microphone (Laptop Only Mode)
# Set to "serial" to use Hybrid Mode (MCU captures audio and sends via Serial to Laptop)
CAPTURE_MODE = "microphone"

# Serial port settings (Only used if CAPTURE_MODE is "serial")
# Change this to match your ATmega32A's COM port (e.g., "COM5")
SERIAL_PORT = "COM7"
SERIAL_BAUDRATE = 230400
