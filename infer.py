from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
	model, tokenizer = AutoModelForCausalLM.from_pretrained("/share_data/base_model"), AutoTokenizer.from_pretrained("/share_data/base_model")
	model.eval()
	text = input("Enter text: ")
	encoded_input = tokenizer(text, return_tensors="pt")
	output = model.generate(**encoded_input, max_length=50, num_return_sequences=5, temperature=0.9)
	for i, sample_output in enumerate(output):
		print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

if __name__ == '__main__':
	main()
