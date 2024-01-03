import openai
import dotenv

'''
Pleco import .txt format
//Header
simp[trad]n\tpiyin\tdefinition
'''

MAX_LINES_PER_BATCH = 15
SYSTEM_MESSAGE = '你是一个乐于助人的助手。'
INSTRUCTIONS = '''Given a list of Chinese sentences, follow this pattern:
<INPUT>
Simplified characters
</INPUT>

Your output:
Simplified characters[Traditional characters]	Pinyin	English translation
    
Here are some specific examples:
<INPUT>
他们告诉我他们会说两种语言。
我还有好多问题要问你呢。
这里没人做那件事。
</INPUT>

Output:
他们告诉我他们会说两种语言。[他們告訴我他們會說兩種語言。]	Tāmen gàosù wǒ tāmen huì shuō liǎng zhǒng yǔyán	They told me they spoke two languages.
我还有好多问题要问你呢[ 我還有好多問題要問你呢 ]	Wǒ hái yǒu hǎoduō wèntí yào wèn nǐ ne	I still have so many questions to ask you
这里没人做那件事。這裡沒人做那件事。]	Zhèlǐ méi rén zuò nà jiàn shì	No one here does that.
    
Here is my input:
<INPUT>
'''
TOKEN_COST_DICT = {
    'gpt-3.5-turbo': {'input': 0.001, 'output': 0.002},
    'gpt-4': {'input': 0.03, 'output': 0.06}
}

def txt2pleco(input_file, output_file, model='gpt-3.5-turbo', known_words=[], header='txt2pleco_import', verbose=False):
    if verbose:
        print('\nTXT2PLECO\n=========', flush=True)
    
    input = open(input_file, 'rb')
    lines = input.readlines()
    batches = [lines[i:i+MAX_LINES_PER_BATCH] for i in range(0, len(lines), MAX_LINES_PER_BATCH)]
    batches = [[line.decode('utf-8') for line in batch] for batch in batches]
    input.close()

    if verbose:
        print('Prompting ChatGPT', flush=True)
    client = openai.OpenAI(
        api_key=dotenv.dotenv_values('secrets.env')['OPENAI_API_KEY']
    )

    total_cost = 0
    output = open(output_file, 'wb+')
    output.write(f'//{header}\n'.encode('utf-8'))
    for i in range(len(batches)):
        if verbose:
            print(f'\tWorking on batch {i+1} of {len(batches)}. Cost so far: ${round(total_cost, 2)}', flush=True)
        prompt = INSTRUCTIONS + ''.join(batches[i]) + '</INPUT>'
        response, usage = prompt_model(client, model, SYSTEM_MESSAGE, prompt)
        total_cost += (usage.prompt_tokens * TOKEN_COST_DICT[model]['input'] / 1000) + \
                     (usage.completion_tokens * TOKEN_COST_DICT[model]['output'] / 1000)
        output.write(response.encode('utf-8'))
        output.write('\n'.encode('utf-8'))

    if verbose:
        print(f'Done prompting ChatGPT. Total cost: ${round(total_cost, 2)}')
    output.close()

def prompt_model(client, model, system_message, prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content, completion.usage

if __name__ == '__main__':
    txt2pleco('transcript.txt', 'pleco_flashcards.txt', header='Qiaohu Ep.1', verbose=True)