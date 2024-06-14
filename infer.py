from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
	model, tokenizer = AutoModelForCausalLM.from_pretrained("/share_data/base_model"), AutoTokenizer.from_pretrained("/share_data/base_model")
	model.eval()
	while True:
		text = input("Enter text: ")
		encoded_input = tokenizer(text, return_tensors="pt")
		output = model.generate(**encoded_input, max_length=100)
		print(output)
		print(tokenizer.decode(output, skip_special_tokens=True))

if __name__ == '__main__':
	main()
