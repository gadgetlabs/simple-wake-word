import struct

import pyaudio
import torch
import torch.nn.functional as F
from g2p_en import G2p
from transformers import WhisperFeatureExtractor, WhisperModel

# ═══════════════════════════════════════════════════════════════════════
# SETUP: Load the components (done once at startup)
# ═══════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0  # seconds
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
SIMILARITY_THRESHOLD = 0.7

# Standard ARPAbet phoneme set (stripped of stress digits)
PHONEMES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]
PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEMES)}

# 1. Grapheme-to-phoneme converter
g2p = G2p()

# 2. Whisper-Tiny encoder (FROZEN - pretrained, ~39M params)
whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
whisper_model.eval()
for param in whisper_model.parameters():
    param.requires_grad = False

feature_extractor = WhisperFeatureExtractor(sampling_rate=SAMPLE_RATE)

# 3. Learned phoneme embedding table (TRAINABLE)
phoneme_embedding_table = torch.nn.Embedding(
    num_embeddings=len(PHONEMES),
    embedding_dim=256,
)

# 4. Text projection head (TRAINABLE - maps phoneme space → shared 128-dim space)
text_proj = torch.nn.Sequential(
    torch.nn.Linear(256, 128),
    torch.nn.LayerNorm(128),
)

# 5. Audio projection head (TRAINABLE - maps whisper space → shared 128-dim space)
audio_proj = torch.nn.Sequential(
    torch.nn.Linear(384, 128),  # Whisper-tiny outputs 384-dim
    torch.nn.LayerNorm(128),
)

# 6. Load trained weights if available
import os
if os.path.exists("wake_word_weights.pt"):
    weights = torch.load("wake_word_weights.pt", weights_only=True)
    phoneme_embedding_table.load_state_dict(weights["phoneme_embedding_table"])
    text_proj.load_state_dict(weights["text_proj"])
    audio_proj.load_state_dict(weights["audio_proj"])
    print("Loaded trained weights from wake_word_weights.pt")


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _record_chunk(stream: pyaudio.Stream) -> torch.Tensor:
    """Read one chunk of audio from the pyaudio stream."""
    data = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
    samples = struct.unpack(f"{CHUNK_SAMPLES}h", data)
    audio = torch.tensor(samples, dtype=torch.float32) / 32768.0  # int16 → [-1, 1]
    return audio


def _audio_to_embedding(audio: torch.Tensor) -> torch.Tensor:
    """Convert raw audio waveform to a normalized 128-dim embedding."""
    inputs = feature_extractor(
        audio.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )
    mel = inputs.input_features  # [1, 80, 3000]

    with torch.no_grad():
        encoder_out = whisper_model.encoder(mel)
        hidden = encoder_out.last_hidden_state  # [1, T, 384]

    pooled = hidden.mean(dim=1)         # [1, 384]
    embedding = audio_proj(pooled)      # [1, 128]
    embedding = F.normalize(embedding, dim=-1)
    return embedding.squeeze(0)         # [128]


def _open_mic_stream() -> tuple[pyaudio.PyAudio, pyaudio.Stream]:
    """Open a pyaudio input stream at 16 kHz mono int16."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SAMPLES,
    )
    return pa, stream


# ═══════════════════════════════════════════════════════════════════════
# ENROLLMENT (happens once when user sets up their wake word)
# ═══════════════════════════════════════════════════════════════════════

def enroll_wake_word(text: str):
    """User types their desired wake word — no audio needed!"""

    # Convert text to phonemes using G2P
    raw_phonemes = g2p(text)
    # "Hey Jarvis" → ['HH', 'EY1', ' ', 'JH', 'AA1', 'R', 'V', 'IH0', 'S']

    # Strip stress digits and skip non-phoneme tokens (spaces, punctuation)
    phonemes = []
    for p in raw_phonemes:
        stripped = p.rstrip("012")
        if stripped in PHONEME_TO_ID:
            phonemes.append(stripped)

    # Look up learned phoneme embeddings
    phoneme_ids = torch.tensor([PHONEME_TO_ID[p] for p in phonemes])
    phoneme_embeds = phoneme_embedding_table(phoneme_ids)
    # Shape: [num_phonemes, 256]

    # Project to shared space (using trained text projection head)
    target_embedding = text_proj(phoneme_embeds.mean(dim=0))
    # Shape: [128]

    # Normalize for cosine similarity
    target_embedding = F.normalize(target_embedding, dim=-1)

    torch.save(target_embedding, "wake_word_embedding.pt")
    print(f"Enrolled wake word: \"{text}\" ({len(phonemes)} phonemes: {phonemes})")
    return target_embedding


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE (runs continuously — this is the real-time loop)
# ═══════════════════════════════════════════════════════════════════════

def listen_for_wake_word(target_embedding: torch.Tensor):
    """Continuously compare audio against enrolled wake word."""
    pa, stream = _open_mic_stream()

    print("Listening for wake word...")
    try:
        while True:
            audio = _record_chunk(stream)
            audio_embedding = _audio_to_embedding(audio)

            similarity = F.cosine_similarity(
                audio_embedding.unsqueeze(0),
                target_embedding.unsqueeze(0),
            )

            score = similarity.item()
            print(f"Similarity: {score:.4f}", end="")

            if score > SIMILARITY_THRESHOLD:
                print(" — Wake word detected!")
                # TODO: trigger your assistant here
            else:
                print()
    except KeyboardInterrupt:
        print("\nStopped listening.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    if os.path.exists("wake_word_embedding.pt"):
        target = torch.load("wake_word_embedding.pt", weights_only=True)
        print("Loaded existing wake word embedding.")
    else:
        target = enroll_wake_word("Hey Amber")

    listen_for_wake_word(target)
