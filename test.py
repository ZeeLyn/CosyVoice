from faster_whisper import WhisperModel

model_size = "large-v2"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("1ebc9f8085f342c2922d1dc7bf8205f5.mp3", beam_size=5,condition_on_previous_text=False)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print(segment)
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))