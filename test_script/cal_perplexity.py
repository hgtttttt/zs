import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, LlamaTokenizer

def load_model_and_tokenizer(model_id, device="cuda"):    
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = 0
    tokenizer.model_max_length = 1e10
    tokenizer.bos_token = '<s>'
    tokenizer.eos_token = '</s>'
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = False
    print("the vocab len of tokenizer is {}".format(len(tokenizer.get_vocab())))
    model.eval()
    return model, tokenizer

def cal_ppl(model_path, test_file, return_detail=False):
    model, tokenizer = load_model_and_tokenizer(model_id=model_path)
    dataset = load_dataset("json", data_files=test_file, split="train")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    print("seq len is {}".format(seq_len))

    max_lengths = [2048*_ for _ in range(1, 17)]
    ppls_2_16k = []
    ppls_16_32k = []
    details = []
    for max_length in max_lengths:
        stride = max_length
        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        ppl = torch.exp(torch.stack(nlls).mean()).item()
        print("perplexity of length {} is {}".format(max_length, ppl))
        if return_detail:
            details.append({"ctx_len":max_length, "ppl":ppl})
        if max_length <= 16384:
            ppls_2_16k.append(ppl)
        else:
            ppls_16_32k.append(ppl)

    final_ppl = 0.7 * sum(ppls_2_16k)/len(ppls_2_16k) + 0.3 * sum(ppls_16_32k)/len(ppls_16_32k)
    print("final ppl is {}".format(final_ppl))
    result = {"final_ppl":final_ppl, "details":details}
    return result

if __name__ == "__main__":
    model_path = "../base_model/"
    test_file = "./demo_ppl.jsonl"
    cal_ppl(model_path=model_path, test_file=test_file)
