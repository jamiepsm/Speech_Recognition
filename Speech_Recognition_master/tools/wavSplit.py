import collections
import contextlib
import wave


def read_wave(path):
    """Liest eine .wav-Datei ein.

    Nimmt den Pfad und gibt (PCM-Audiodaten, Abtastrate) zurück.
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        frames = wf.getnframes()
        pcm_data = wf.readframes(frames)
        duration = frames / sample_rate
        return pcm_data, sample_rate, duration


def write_wave(path, audio, sample_rate):
    """Schreibt eine .wav-Datei.

    Nimmt Pfad, PCM-Audiodaten und Abtastrate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Repräsentiert einen "Frame" von Audiodaten."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generiert Audiorahmen aus PCM-Audiodaten.

    Nimmt die gewünschte Rahmendauer in Millisekunden, die PCM-Daten und
    die Abtastrate.

    Gibt Frames der angeforderten Dauer zurück.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filtert nicht-stimmhafte Audioclips heraus.

    Eine Funktion namens filter_voiced_audio_frames nimmt eine Instanz von
    webrtcvad.Vad und eine Quelle von Audioclip-Frames entgegen und gibt nur
    die stimmliche Audio zurück.

    Die Funktion verwendet einen gepolsterten, gleitenden Fensteralgorithmus über
    den Audioclip-Frames. Wenn mehr als 90% der Frames im Fenster stimmhaft
    sind (wie vom VAD gemeldet), wird der Collector ausgelöst und
    beginnt, Audioclip-Frames zurückzugeben. Dann wartet der Collector, bis 90% der
    Frames im Fenster unstimmhaft sind, um das Zurückgeben zu beenden.

    Das Fenster ist vorne und hinten gepolstert, um eine kleine Menge an Stille
    oder den Beginn/Ende der Sprache um die stimmhaften Frames herum bereitzustellen.

    Argumente:

    sample_rate - Die Audio-Sample-Rate in Hz.
    frame_duration_ms - Die Frame-Dauer in Millisekunden.
    padding_duration_ms - Die Dauer, um das Fenster zu polstern, in Millisekunden.
    vad - Eine Instanz von webrtcvad.Vad.
    frames - eine Quelle von Audioclip-Frames (Sequenz oder Generator).
    Returns: Ein Generator, der PCM-Audioclips-Daten zurückgibt.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # Wir verwenden eine deque für unser gleitendes Fenster / Ringpuffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # Wir haben zwei Zustände: AUSGELÖST und NICHTAUSGELÖST. Wir starten im
    # NICHTAUSGELÖSTEN Zustand.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # Wenn wir NICHTAUSGELÖST sind und mehr als 90% der Frames im
            # Ringpuffer stimmhafte Frames sind, dann wechseln wir in den
            # AUSGELÖSTEN Zustand.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # Wir möchten alle Audio-Daten zurückgeben, die wir von nun an
                # sehen, bis wir NICHTAUSGELÖST sind, aber wir müssen mit dem
                # Audio beginnen, das bereits im Ringpuffer enthalten ist.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # Wir befinden uns im AUSGELÖSTEN Zustand, sammeln also die
            # Audiodaten und fügen sie dem Ringpuffer hinzu.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # Wenn mehr als 90% der Frames im Ringpuffer unstimmhaft sind,
            # wechseln wir zu NICHTAUSGELÖSTEN und geben das gesammelte Audio
            # zurück.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        pass
    # Wenn wir übrig gebliebene stimmhafte Audio haben, wenn uns die
    # Eingabe ausgeht, geben wir sie zurück.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

