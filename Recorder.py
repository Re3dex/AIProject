import sounddevice as sd
import wavio as wv

class AudioRecorder:
    counter_file = "Recordings/recording_counter.txt"

    def __init__(self, freq=44100, duration=20):
        self.freq = freq
        self.duration = duration
        self.recording = None

        # Load the recording counter from the file or set it to 1 if the file doesn't exist
        try:
            with open(AudioRecorder.counter_file, "r") as file:
                self.current_counter = int(file.read().strip())
        except FileNotFoundError:
            self.current_counter = 1

    def record_and_save(self, file_name="Recordings/recording"):
        print(f"Recording Now ({self.duration} sec)")
        self.recording = sd.rec(int(self.duration * self.freq), samplerate=self.freq, channels=2)
        sd.wait()

        file_path = f"{file_name}{self.current_counter}.wav"
        wv.write(file_path, self.recording, self.freq, sampwidth=2)
        print(f"Recording saved as {file_path}")

        # Increment the counter for the next recording
        self.current_counter += 1

        # Save the updated counter to the file
        with open(AudioRecorder.counter_file, "w") as file:
            file.write(str(self.current_counter))
    
    def test_record(self, file_name="recording.wav"):
        print(f"Recording Now ({self.duration} sec)")
        self.recording = sd.rec(int(self.duration * self.freq), samplerate=self.freq, channels=2)
        sd.wait()
        
        wv.write(file_name, self.recording, self.freq, sampwidth=2)
        print(f"Recording saved as {file_name}")

if __name__ == "__main__":
    recorder1 = AudioRecorder()
    recorder1.record_and_save()
