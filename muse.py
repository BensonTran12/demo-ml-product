from pylsl import StreamInlet, resolve_byprop
import numpy as np

print("Looking for a Muse EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=5)
# instead of streaming to terminal you'd want to stream too sckit.py folder
if streams:
    inlet = StreamInlet(streams[0])
    print("Receiving data...")
    while True:
        sample, timestamp = inlet.pull_sample()
        print(f"Timestamp: {timestamp}, EEG: {sample}")
else:
    print("No EEG stream found.")
