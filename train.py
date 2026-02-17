import re
import threading
import queue

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from g2p_en import G2p
from transformers import WhisperFeatureExtractor, WhisperModel

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

BATCH_SIZE = 128
NUM_STEPS = 5000
LR = 1e-3
TEMPERATURE = 0.07
LOG_EVERY = 100
SAMPLE_RATE = 16000
CHECKPOINT_PATH = "wake_word_weights.pt"
PREFETCH_BATCHES = 4  # number of batches to prepare ahead on CPU

PHONEMES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]
PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEMES)}

# ═══════════════════════════════════════════════════════════════════════
# MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Frozen Whisper encoder
whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny").to(device)
whisper_model.eval()
for param in whisper_model.parameters():
    param.requires_grad = False

feature_extractor = WhisperFeatureExtractor(sampling_rate=SAMPLE_RATE)

# Trainable modules
phoneme_embedding_table = nn.Embedding(len(PHONEMES), 256).to(device)
text_proj = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128)).to(device)
audio_proj = nn.Sequential(nn.Linear(384, 128), nn.LayerNorm(128)).to(device)

# ═══════════════════════════════════════════════════════════════════════
# G2P SETUP
# ═══════════════════════════════════════════════════════════════════════

g2p = G2p()


def text_to_phoneme_ids(text: str) -> list[int] | None:
    """Convert text to a list of phoneme IDs, or None if empty."""
    raw = g2p(text)
    ids = []
    for p in raw:
        stripped = re.sub(r"[012]", "", p)
        if stripped in PHONEME_TO_ID:
            ids.append(PHONEME_TO_ID[stripped])
    return ids if ids else None


# ═══════════════════════════════════════════════════════════════════════
# DATASET WITH PREFETCHING
# ═══════════════════════════════════════════════════════════════════════

print("Loading LibriSpeech dataset (streaming)...")

dataset = load_dataset(
    "openslr/librispeech_asr",
    "clean",
    split="train.360",
    streaming=True,
)


def collate_batch(iterator):
    """Pull samples from the streaming iterator until we have a full batch.
    Returns mel features (on CPU) and phoneme ID lists."""
    audio_arrays = []
    phoneme_id_lists = []

    for sample in iterator:
        text = sample.get("text", "")
        if not text:
            continue

        ids = text_to_phoneme_ids(text)
        if ids is None:
            continue

        audio_array = sample["audio"]["array"]
        if len(audio_array) < SAMPLE_RATE:  # skip very short clips
            continue

        audio_arrays.append(audio_array)
        phoneme_id_lists.append(ids)

        if len(audio_arrays) == BATCH_SIZE:
            break

    if len(audio_arrays) < 2:
        return None, None

    # Do mel extraction on CPU in the background thread
    mel_inputs = feature_extractor(
        audio_arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        padding="max_length", truncation=True,
    )
    return mel_inputs.input_features, phoneme_id_lists


def prefetch_worker(iterator, q):
    """Background thread that fills a queue with prepared batches."""
    for _ in range(NUM_STEPS):
        mel, phoneme_ids = collate_batch(iterator)
        if mel is None:
            break
        q.put((mel, phoneme_ids))
    q.put(None)  # sentinel


batch_queue = queue.Queue(maxsize=PREFETCH_BATCHES)
data_iter = iter(dataset)

# Start prefetching in background
prefetch_thread = threading.Thread(
    target=prefetch_worker, args=(data_iter, batch_queue), daemon=True
)
prefetch_thread.start()

# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

optimizer = torch.optim.AdamW(
    list(phoneme_embedding_table.parameters())
    + list(text_proj.parameters())
    + list(audio_proj.parameters()),
    lr=LR,
)

print(f"Training for {NUM_STEPS} steps, batch size {BATCH_SIZE}, device={device}...")

for step in range(1, NUM_STEPS + 1):
    batch = batch_queue.get()
    if batch is None:
        print("Ran out of data, stopping early.")
        break

    mel, phoneme_id_lists = batch
    mel = mel.to(device)  # [B, 80, 3000]
    B = mel.shape[0]

    # --- Audio path (frozen encoder + trainable projection) ---
    with torch.no_grad():
        encoder_out = whisper_model.encoder(mel)
        hidden = encoder_out.last_hidden_state  # [B, T, 384]

    audio_pooled = hidden.mean(dim=1)  # [B, 384]
    audio_emb = F.normalize(audio_proj(audio_pooled), dim=-1)  # [B, 128]

    # --- Text path (trainable embedding + projection) ---
    text_embs = []
    for ids in phoneme_id_lists:
        id_tensor = torch.tensor(ids, device=device)
        emb = phoneme_embedding_table(id_tensor).mean(dim=0)  # [256]
        emb = text_proj(emb)  # [128]
        text_embs.append(emb)
    text_emb = F.normalize(torch.stack(text_embs), dim=-1)  # [B, 128]

    # --- InfoNCE loss (symmetric) ---
    logits = (audio_emb @ text_emb.T) / TEMPERATURE  # [B, B]
    labels = torch.arange(B, device=device)
    loss_a2t = F.cross_entropy(logits, labels)
    loss_t2a = F.cross_entropy(logits.T, labels)
    loss = (loss_a2t + loss_t2a) / 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % LOG_EVERY == 0 or step == 1:
        print(f"Step {step:>5d}/{NUM_STEPS}  loss={loss.item():.4f}")

# ═══════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════

torch.save(
    {
        "phoneme_embedding_table": phoneme_embedding_table.state_dict(),
        "text_proj": text_proj.state_dict(),
        "audio_proj": audio_proj.state_dict(),
    },
    CHECKPOINT_PATH,
)
print(f"Saved checkpoint to {CHECKPOINT_PATH}")
