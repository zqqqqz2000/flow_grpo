from torch.utils.data import Dataset


class SceneDataset(Dataset):
    def __init__(self, file_path):
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Flatten all scenes from different difficulty levels
        self.scenes = []
        for difficulty, scenes_list in data.items():
            for scene in scenes_list:
                # Add difficulty info to each scene
                scene["difficulty"] = difficulty
                self.scenes.append(scene)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        return {
            "prompt": scene["prompt"],
            "qa": scene["qa"],
            "difficulty": scene["difficulty"],
        }
