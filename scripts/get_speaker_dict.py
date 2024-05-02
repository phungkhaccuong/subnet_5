
from tqdm import tqdm
import os
from pathlib import Path
import json

root_dir = __file__.split("scripts")[0]
index_name = "eth_denver_v1"

def get_speaker_dict():
    """Index documents in Elasticsearch"""

    dataset_dir = root_dir + "datasets/eth_denver_dataset"
    dataset_path = Path(dataset_dir)

    num_files = len(list(dataset_path.glob("*.json")))
    print(f"Indexing {num_files} files in {dataset_dir}")

    speaker_dict = {}
    for doc_file in tqdm(
            dataset_path.glob("*.json"), total=num_files, desc="Indexing docs"
    ):
        with open(doc_file, "r") as f:
            doc = json.load(f)
            speaker = doc['speaker']
            episode_id = doc['episode_id']
            if speaker in speaker_dict:
                speaker_dict[speaker].append(episode_id)
            else:
                speaker_dict[speaker] = [episode_id]

    # Convert sets to lists
    speaker_episode_dict = {speaker: list(set(episode_ids)) for speaker, episode_ids in speaker_episode_dict.items()}

    # Save speaker_episode_dict to a JSON file
    with open('speaker_dict.json', 'w') as json_file:
        json.dump(speaker_dict, json_file)

    print("JSON file has been created successfully!")
    print(speaker_dict)



if __name__ == "__main__":
    get_speaker_dict()



