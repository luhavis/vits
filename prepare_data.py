from glob import glob
from faster_whisper import WhisperModel


def make_train_text_file(train_text_file: str, wav_dir: str):
    wav_files = glob(f"{wav_dir}/*.wav")

    model = WhisperModel(
        model_size_or_path="large-v2",
        device="cuda",
        compute_type="float16",
    )

    with open(train_text_file, "w", encoding="utf-8") as f:
        
        for wav_file in wav_files:
            segments, info = model.transcribe(
                wav_file,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500
                ),
            )

            text = " ".join([segment.text for segment in segments]).strip()

            print(f"{wav_file}|0|{text}")
            f.writelines(f"{wav_file}|0|{text}\n")




if __name__ == "__main__":
    make_train_text_file("filelists/train.txt", "datasets")

