from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
MODEL_NAME = 'google/t5-small-ssm-nq'


def generate_response(question: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    tokenized_question = tokenizer(question, return_tensors='pt').input_ids

    tokenized_answer = model.generate(tokenized_question, max_new_tokens=50)[0]

    answer = tokenizer.decode(tokenized_answer, skip_special_tokens=True)

    print(f"Question: {question}\nResponse: {answer}")
    return answer


if __name__ == '__main__':
    generate_response('What should i do to make the things better?')