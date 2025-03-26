import sounddevice as sd
import librosa

# Load the audio file with its original sampling rate
audio_file = "Jupiter.wav"
y, sr = librosa.load(audio_file, sr=None)

# Speed and pitch manipulation parameters
current_speed = 1.0  # 1.0 = original speed
current_pitch = 0    # 0 = original pitch

# Apply time stretching if needed
if current_speed != 1.0:
    y = librosa.effects.time_stretch(y, rate=current_speed)

# Apply pitch shifting if needed
if current_pitch != 0:
    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=current_pitch)

# Play the processed audio
sd.play(y, sr)
sd.wait()  # Wait until playback is complete
