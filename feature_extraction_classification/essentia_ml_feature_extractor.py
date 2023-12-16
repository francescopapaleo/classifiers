import os
import json
import essentia
import essentia.standard as estd
import numpy as np

from essentia.standard import MonoLoader, RhythmExtractor2013, Danceability, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
from pathlib import Path
from tqdm import tqdm

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from style_labels import style_400_keys # This is a list of 400 style labels based on the Discogs dataset


class EssentiaMLFeatureExtractor:
    '''Class to extract features from audio files using Essentia pre-trained models'''

    def __init__(self):
        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
        self.model_genre = TensorflowPredict2D(graphFilename="models/genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")

    def tempo_dance(self, path_to_file):
        audio = MonoLoader(filename=path_to_file, sampleRate=44100)()
        bpm, _, _, _, _ = RhythmExtractor2013()(audio)
        danceability, dfa = Danceability()(audio)
        return bpm, danceability

    def audio_16(self, path_to_file):
        audio_load_16 = MonoLoader(filename=path_to_file, sampleRate=16000)()
        return audio_load_16

    def genre_classification(self, audio_load_16):
        embeddings = self.embedding_model(audio_load_16)
        predictions = self.model_genre(embeddings)
        
        predictions_mean = np.mean(predictions, axis=0, keepdims=True)
        predictions_list = list(predictions_mean.flatten())
        style_zip = dict(zip(style_400_keys, predictions_list))
        return style_zip


    def compute_descriptors(self, file_path):
        descriptors_dict = {}

        rel_path = file_path
        bpm, danceability = self.tempo_dance(file_path)
        audio_16 = self.audio_16(file_path)
        genre_zip = self.genre_classification(audio_16)
        
        descriptors_dict = {
            'file_path': rel_path,
            'bpm': str(bpm),
            'danceability': str(danceability),
            'style_predictions': str(genre_zip),
        }
        return descriptors_dict


def main():

    DATA_DIR = "../audio/raw/"
    OUTPUT_FILE = "feature_extractor_output.json"
    
    all_files_list = []
    # Get all the files in the data directory
    for path in Path(DATA_DIR).rglob('*.wav'):
        all_files_list.append(str(path))

    files_to_process = all_files_list[:]

    print("Number of files to process: {}".format(len(files_to_process)))
    
    # Instantiate the class
    feature_extractor = EssentiaMLFeatureExtractor()

    # compute the descriptors and save them to a json file at each iteration, one dict per line
    with open(OUTPUT_FILE, "w") as f:
        f.write("[")                # add opening bracket
        for i, file_path in enumerate(tqdm(files_to_process)):
            if i != 0:
                f.write(",")        # add comma between dicts
            features = feature_extractor.compute_descriptors(file_path)
            json.dump(features, f, indent=0)
        f.write("]")                # add closing bracket

if __name__ == "__main__":
   
    main()
