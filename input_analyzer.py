with open('resources/asap_prompt_1.txt', 'r', encoding='utf-8') as file:
    max_length = 0
    min_length = 1024
    for line in file:
        if line is '\n':
            continue
        if len(line) > max_length:
            max_length = len(line)
        if len(line) < min_length:
            min_length = len(line)
    print('Max sentence length in this prompt is '+str(max_length))
    print('Min sentence length in this prompt is '+str(min_length))
    file.close()
