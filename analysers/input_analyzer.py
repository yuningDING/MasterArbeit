for i in range(1, 11):
    with open('resources/asap_prompt_'+str(i)+'.txt', 'r', encoding='utf-8') as file:
        max_length = 0
        sum_length = 0
        count_line = 0
        min_length = 1024
        for line in file:
            token_list = line.split(" ")
            if len(token_list) > max_length:
                max_length = len(token_list)
                sum_length += len(token_list)
                count_line += 1
            if len(token_list) < min_length:
                min_length = len(token_list)
                sum_length += len(token_list)
                count_line += 1
        print('Min sentence length in this prompt' + str(i) + ' is ' + str(min_length))
        print('Max sentence length in this prompt' + str(i) + ' is ' + str(max_length))
        print('Average sentence length in prompt ' + str(i) + ' is ' + str(sum_length/count_line))
        print()
        file.close()
