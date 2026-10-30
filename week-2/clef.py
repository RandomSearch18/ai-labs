import sys

import torch
from huggingface_hub import snapshot_download

path = snapshot_download("Cloudflare/clef-flash")
sys.path.insert(0, path)
from joint_schema_model import collate_records, encode_record, load_release_model

model, processor = load_release_model(path, device="cuda")

record = {
    "state": {
        "invoice": {
            "vendor": "Acme",
            "total": 1250.0,
            "currency": "USD",
            "status": "overdue",
        }
    },
    "questions": {
        "status": {
            "type": "choice",
            "instructions": "What is the invoice status?",
            "criteria": {
                "paid": "Invoice is paid.",
                "overdue": "Invoice is past due.",
                "draft": "Not sent.",
            },
        },
        "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
    },
}

encoded = encode_record(processor.tokenizer, record, processor=processor)
batch = collate_records(
    [encoded], processor.tokenizer.pad_token_id, torch.device("cuda")
)
with torch.inference_mode():
    logits = model(batch)[0]

for question, question_logits in zip(encoded.questions, logits):
    probabilities = question_logits.float().softmax(-1).tolist()
    print(question.question_id, dict(zip(question.option_ids, probabilities)))
