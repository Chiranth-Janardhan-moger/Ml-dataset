import os
import torch
from transformers import WhisperFeatureExtractor
from src.config import Config
from src.features import load_and_preprocess_audio, extract_mel_spectrogram_image, extract_sequential_features
from src.models.fusion_model import SilentCryDecoder

class InferencePipeline:
    def __init__(self, checkpoint_path=None, device=Config.DEVICE):
        self.device = device
        self.whisper_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        self.model = SilentCryDecoder().to(self.device)
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(Config.SAVE_DIR, "best_model.pth")
            
        if os.path.exists(checkpoint_path):
            print(f"Loading model checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"Warning: Checkpoint '{checkpoint_path}' not found. Using uninitialized weights.")
            
        self.model.eval()
        
    def predict(self, audio_file_path):
        """
        Runs single-file inference, returns predictions probabilities and label.
        """
        # 1. Preprocess raw audio
        y, sr = load_and_preprocess_audio(audio_file_path)
        
        # 2. Extract features
        visual_feat = extract_mel_spectrogram_image(y, sr).unsqueeze(0).to(self.device) # (1, 3, 224, 224)
        seq_feat = extract_sequential_features(y, sr).unsqueeze(0).to(self.device) # (1, TIME_STEPS, 125)
        
        whisper_inputs = self.whisper_extractor(y, sampling_rate=sr, return_tensors="pt")
        whisper_feat = whisper_inputs.input_features.to(self.device) # (1, 80, 3000)
        
        # 3. Model Forward
        with torch.no_grad():
            logits = self.model(visual_feat, seq_feat, whisper_feat)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
        pred_idx = int(probs.argmax())
        pred_label = Config.LABEL_MAP[pred_idx]
        pred_prob = float(probs[pred_idx])
        
        # Structure the response
        class_probabilities = {
            Config.LABEL_MAP[i]: float(probs[i]) for i in range(Config.NUM_CLASSES)
        }
        
        # Recommendations & Action Items based on the cry reasons
        recommendations = {
            "Hunger": "The baby is hungry. Feed them with milk or appropriate food. Check if it's been more than 2-3 hours since the last feeding.",
            "Pain (Belly Pain)": "The baby is in pain, likely gas or colic. Try the 'baby bicycle' leg exercise, gently massage the tummy clockwise, or burp the baby. Hold them close to soothe them.",
            "Sleepy (Tired)": "The baby is tired and needs sleep. Dim the lights, play soft white noise or a lullaby, rock them gently, and put them in a quiet, comfortable sleeping environment.",
            "Discomfort": "The baby feels uncomfortable. Check if their diaper is wet/soiled, if their clothing is too tight, or if they are feeling too hot or too cold. Adjust clothing/environment accordingly.",
            "Burping": "The baby needs to burp, likely due to trapped air. Hold the baby upright against your chest with their chin resting on your shoulder and gently pat/rub their back."
        }
        
        return {
            "class_index": pred_idx,
            "predicted_reason": pred_label,
            "confidence": pred_prob,
            "all_probabilities": class_probabilities,
            "recommendation": recommendations.get(pred_label, "Observe baby carefully and consult a pediatrician if crying persists.")
        }

if __name__ == "__main__":
    # Test pipeline on a sample audio file if exists
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        pipeline = InferencePipeline()
        result = pipeline.predict(test_file)
        print("Inference Result:")
        print("-" * 30)
        print(f"Predicted Reason: {result['predicted_reason']} ({result['confidence']*100:.2f}%)")
        print("Detailed Probabilities:")
        for label, prob in result['all_probabilities'].items():
            print(f"  {label}: {prob*100:.2f}%")
        print(f"Recommendation: {result['recommendation']}")
    else:
        print("Usage: python src/inference.py <path_to_wav_file>")
