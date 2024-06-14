import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, LlamaTokenizer

# 模型prompt应遵循规范 <human>与<bot>的角色名称，以下是一个示例
template = """<human>: 你是一个智能AI助手,请根据以下文档回答问题
{}
给你的问题是：{}
请作答
<bot>: """

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

def cal_coef(model_path, test_file, return_detail=False):
    model, tokenizer = load_model_and_tokenizer(model_id=model_path)
    dataset = load_dataset("json", data_files=test_file, split="train")
    right = 0
    details = []
    for line in tqdm(dataset):
        prompt = template.format(line["context"], line["question"])
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            generate_tokens = model.generate(**inputs,
                                                    max_new_tokens=20,
                                                    do_sample=False,
                                                    eos_token_id=[tokenizer.bos_token_id, tokenizer.eos_token_id])[0]
        model_answer = tokenizer.decode(generate_tokens[len(inputs["input_ids"][0]):], True)
        if return_detail:
            details.append({"answer":line["answer"], "text_info":line["info"], "output":model_answer})
        if line['answer'] in model_answer:
            right += 1
    acc = right / len(dataset)
    coef = round(1 + (1-acc) * 0.5, 2)
    result = {"acc": round(acc, 2), "coef": coef, "details":details}
    print("acc is {}, coef is {}".format(round(acc, 2), coef))
    return result

if __name__ == "__main__":
    model_path = "../base_model"
    test_file = "./demo_needle.jsonl"
    cal_coef(model_path=model_path, test_file=test_file)