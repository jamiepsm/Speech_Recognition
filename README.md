Eine Linux-basierte Anwendung, die Audiodateien von Englisch oder Deutsch in Text überstezten kann. Anschließend kann sie auch die Stimmung im Text erkennen. Funktioniert komplett offline!

0. Voraussetzungen

Richten Sie Ihre Umgebung ein.

Wenn virtuelle Umgebung auf Ihrem Linux-System nicht installiert ist:
```
sudo apt install virtualenv
```

Navigieren Sie zum Transcriber-Verzeichnis (oder erstellen Sie eins):
```
mkdir transcriber
cd transcriber
```

Klonen Sie diesen Git-Kontent:
```
git clone https://github.com/jamiepsm/Speech_Recognition.git
cd Speech_Recognition
```

Initialisieren und aktivieren Sie eine virtuelle Umgebung:
```
virtualenv -p python3 transcriber
source transcriber/bin/activate
```
oder

```
mkvirtualenv transcriber
```
Installieren Sie die Voraussetzungen in der virtuellen Umgebung:
```
(transcriber) pip3 install -r requirements/requirements.txt
```
Für die Sentimentanalyse verwenden wir corpora von TextBlob.

Führen Sie den folgenden Befehl aus, um sie herunterzuladen:
```
python -m textblob.download_corpora
```
1. Modelle und Beispieldateien herunterladen
Die Modelldateien sind zu groß, um sie auf GitHub hochzuladen. Laden Sie sie daher von hier herunter.

Sobald die Modelldateien heruntergeladen wurden, verschieben Sie sie in die richtige Verzeichnisstruktur, damit die Hauptdatei sie lesen kann.

Verzeichnisstruktur:

```
~(home/user/)
 |----Speech_Recognition_and_Emotion_Detection_in_English_and_German
     |----audio
          |----english
             |----audio_file_sample.wav
          |----german
             |----audio_file_sample.wav
     |----models
          |----english
             |----alphabet.txt
             |----lm.binary
             |----output_graph.pb
             |----trie
          |----german
             |----alphabet.txt
             |----lm.binary
             |----output_graph.pb
             |----trie
     |----requirements
          |----requirements.txt
     |----tools
          |----wavSplit.py
          |----wavTranscriber.py
     |----transcriber_gui.py 
```
2. Transkription / Arbeiten mit der GUI
Führen Sie einfach den folgenden Befehl aus, um die GUI zu starten:
```
python3 transcriber_gui.py
```
Schritte zur Verwendung der GUI:

Wählen Sie die Sprache: Englisch oder Deutsch
Wählen Sie die Eingabe: Mikrofon oder Datei-Upload
Durchsuchen Sie die WAV-Datei, wenn der Datei-Upload ausgewählt ist
Klicken Sie auf 'Start speaking' für das Mikrofon oder 'Transcribe wav' für den Datei-Upload
Die Ausgabe von Text und Stimmung wird im Transkriptionsfenster angezeigt.

Genießen! :')

Hinweis: Wenn die GUI abstürzt, denken Sie daran, diese Prozedur zu befolgen: Jedes Mal, wenn Sie zwischen 'Mikrofon' und 'Datei-Upload' wechseln möchten, klicken Sie immer auf diese Reihenfolge - 'Sprache' -> 'Mikrofon / Datei' -> 'Start Speaking / Transcribe WAV'.

Wenn Sie von Datei-Upload zu Mikrofon (oder Mikrofon zu Datei) wechseln möchten, klicken Sie immer zuerst auf die Sprache, dann auf Mikrofon und dann auf Start Speaking.


Ich habe hauptsächlich Mozilla DeepSpeech verwendet - https://github.com/mozilla/DeepSpeech

Englische Sentimentanalyse - https://www.liip.ch/en/blog/sentiment-detection-with-keras-word-embeddings-and-lstm-deep-learning-networks

Deutsche Sentimentanalyse - https://textblob-de.readthedocs.io
