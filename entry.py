import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from torch import nn
import tqdm


def load(model_path="/mnt/d/Studio/Python/models/DeepSeek-R1-Distill-Qwen-1.5B"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    enc = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    return model, enc


def evaluate(model, enc):
    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
    model.seqlen = 2048
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    results = {"ppl": ppl.item()}
    return results


if __name__ == "__main__":
    model, enc = load()
    evaluate(model, enc)
