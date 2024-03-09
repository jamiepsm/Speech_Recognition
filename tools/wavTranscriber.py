import glob
import webrtcvad
import logging
from tools import wavSplit
from deepspeech import Model
from timeit import default_timer as timer

'''
Lädt das vortrainierte Modell in den Speicher
@param models: Ausgabe-Graph-Protokolldatei
@param alphabet: Alphabet.txt-Datei
@param lm: Sprachmodell-Datei
@param trie: Trie-Datei

@Rückgabewert
Gibt eine Liste zurück [DeepSpeech-Objekt, Ladezeit des Modells, Ladezeit des Sprachmodells]
'''
def load_model(models, alphabet, lm, trie):
    N_FEATURES = 26
    N_CONTEXT = 9
    BEAM_WIDTH = 500
    LM_ALPHA = 0.75
    LM_BETA = 1.85

    model_load_start = timer()
    ds = Model(models, N_FEATURES, N_CONTEXT, alphabet, BEAM_WIDTH)
    model_load_end = timer() - model_load_start
    logging.debug("Modell geladen in %0.3fs." % (model_load_end))

    lm_load_start = timer()
    ds.enableDecoderWithLM(alphabet, lm, trie, LM_ALPHA, LM_BETA)
    lm_load_end = timer() - lm_load_start
    logging.debug('Sprachmodell geladen in %0.3fs.' % (lm_load_end))

    return [ds, model_load_end, lm_load_end]

'''
Führt Inferenz auf der Eingabe-Audiodatei durch
@param ds: DeepSpeech-Objekt
@param audio: Eingabeaudio für die Inferenz
@param fs: Abtastrate der Eingabe-Audiodatei

@Rückgabewert:
Gibt eine Liste zurück [Inferenz, Inferenzzeit, Audiolänge]

'''
def stt(ds, audio, fs):
    inference_time = 0.0
    audio_length = len(audio) * (1 / 16000)

    # Führt Deepspeech aus
    logging.debug('Führe Inferenz durch...')
    inference_start = timer()
    output = ds.stt(audio, fs)
    inference_end = timer() - inference_start
    inference_time += inference_end
    logging.debug('Inferenz dauerte %0.3fs für eine Audiodatei von %0.3fs.' % (inference_end, audio_length))

    return [output, inference_time]

'''
Löst den Verzeichnispfad für die Modelle auf und ruft jeden von ihnen ab.
@param dirName: Pfad zum Verzeichnis mit vortrainierten Modellen

@Rückgabewert:
Gibt ein Tupel zurück, das jede der Modelldateien enthält (pb, Alphabet, lm und Trie)
'''
def resolve_models(dirName):
    pb = glob.glob(dirName + "/*.pb")[0]
    logging.debug("Modell gefunden: %s" % pb)

    alphabet = glob.glob(dirName + "/alphabet.txt")[0]
    logging.debug("Alphabet gefunden: %s" % alphabet)

    lm = glob.glob(dirName + "/lm.binary")[0]
    trie = glob.glob(dirName + "/trie")[0]
    logging.debug("Sprachmodell gefunden: %s" % lm)
    logging.debug("Trie gefunden: %s" % trie)

    return pb, alphabet, lm, trie

'''
Generiert VAD-Segmente. Filtert nicht-stimmhafte Audioclip-Frames heraus.
@param waveFile: Eingabewav-Datei, auf der VAD ausgeführt werden soll.

@Rückgabewert:
Gibt ein Tupel zurück aus
    segments: ein ByteArray von mehreren kleineren Audiorahmen
              (Das längere Audio in mehrere kleinere aufgeteilt)
    sample_rate: Abtastrate der Eingabe-Audiodatei
    audio_length: Dauer der Eingabe-Audiodatei

'''
def vad_segment_generator(wavFile, aggressiveness):
    logging.debug("Datei gefunden @: %s" % (wavFile))
    audio, sample_rate, audio_length = wavSplit.read_wave(wavFile)
    assert sample_rate == 16000, "Nur 16000Hz-Eingabewav-Dateien werden derzeit unterstützt!"
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = wavSplit.frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = wavSplit.vad_collector(sample_rate, 30, 300, vad, frames)

    return segments, sample_rate, audio_length