import os
import sys
import torch
import yaml
from transformers import AutoModel
from huggingface_hub import hf_hub_download
from .utils import preprocess_image, preprocess_batch

class AdaFaceVerifier:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.repo_id = self.config['model']['repo_id']
        self.threshold = self.config['verification']['threshold']
        self.device = torch.device(self.config['model']['device'] if self.config['model']['device'] != 'auto' 
                                   else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self):
        cache_dir = os.path.expanduser("~/.cvlface_cache")
        repo_slug = self.repo_id.replace("/", "_")
        local_path = os.path.join(cache_dir, repo_slug)
        
        if not os.path.exists(local_path):
            print(f"Downloading model {self.repo_id} to local cache...")
            self._download_cvlface_repo(local_path)

        # The CVLFace "hack" to load relative imports
        sys.path.insert(0, local_path)
        cwd = os.getcwd()
        os.chdir(local_path)
        
        try:
            model = AutoModel.from_pretrained(local_path, trust_remote_code=True)
        finally:
            os.chdir(cwd)
            if local_path in sys.path:
                sys.path.remove(local_path)
        
        return model

    def _download_cvlface_repo(self, path):
        os.makedirs(path, exist_ok=True)
        # Download essential files
        files = ['config.json', 'wrapper.py', 'model.safetensors', 'model.yaml', 'v1_ir101.yaml', '__init__.py', 'utils.py']
        # Also need the whole directory for relative imports
        from huggingface_hub import HfApi
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=self.repo_id)
        for file in repo_files:
            hf_hub_download(self.repo_id, file, local_dir=path, local_dir_use_symlinks=False)

    @torch.no_grad()
    def get_embedding(self, img_path_or_bgr):
        """Extract L2-Normalized embedding for a single face."""
        tensor = preprocess_image(img_path_or_bgr)
        if tensor is None:
            return None
        emb = self.model(tensor)
        # Force L2 Normalization
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.view(-1)

    @torch.no_grad()
    def get_embeddings_batch(self, imgs_list):
        """
        EXTRACT NORMALIZED EMBEDDINGS IN BATCH.
        """
        batch_tensor = preprocess_batch(imgs_list)
        if batch_tensor is None:
            return []
        
        # Infer whole batch
        embs = self.model(batch_tensor)
        # Force L2 Normalization for the whole batch
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        
        return [embs[i] for i in range(embs.shape[0])]

    def verify(self, img1, img2):
        emb1 = self.get_embedding(img1)
        emb2 = self.get_embedding(img2)
        
        if emb1 is None or emb2 is None:
            return False, 0.0
            
        similarity = torch.dot(emb1, emb2).item()
        return similarity > self.threshold, similarity
