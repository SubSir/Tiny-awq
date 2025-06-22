import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from torch import nn
import tqdm

from pseudo import Mx


def load(model_path="/mnt/d/Studio/Python/models/DeepSeek-R1-Distill-Qwen-1.5B"):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
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


@torch.no_grad()
def chat(
    model,
    tokenizer,
    prompt="为什么陨石总是精准砸进陨石坑？是宇宙在瞄准吗？",
    max_length=512,
    temperature=0.7,
    top_p=0.9,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_prompt = f"你是一个有帮助的AI助手。请用自然、简洁的方式回答以下问题:\n\n用户: {prompt}\n助手:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            attention_mask=inputs.attention_mask,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


if __name__ == "__main__":
    mx = Mx()
    model, enc = load()
    model = mx.pseudo_quantize(model)
    evaluate(model, enc)
