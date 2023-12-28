import openai
import dotenv

'''
Pleco import .txt format
//Header
simp[trad]\tpinyin\tdefinition
'''

def txt2pleco(input_file, output_file, known_words=[], header='txt2pleco_import', ai_curate=True, verbose=False):
    if verbose:
        print('\nTXT2PLECO\n========', flush=True)
    
    system_message = 'You are a helpful assistant. 你是一个乐于助人的助手。'
    instructions = '''I have a list of Chinese subtitles. 
    I want you to go through each line and generate the simplified and traditional
    character versions, the pinyin, and the English definition.  '''
    if ai_curate:
        instructions += ' Additionally, some of the subtitles may be entirely incorrect. Remove any lines that are nonsensical.'
    instructions += '''\nI want the output to follow this pattern:
    Input:
    Simplified characters

    Output:
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

    input = open(input_file, 'rb')
    lines = input.read().decode('utf-8')
    instructions += lines + '\n</INPUT>'
    
    if verbose:
        print('Prompting ChatGPT', flush=True)
    client = openai.OpenAI(
        api_key=dotenv.dotenv_values('secrets.env')['OPENAI_API_KEY']
    )
    response = prompt_model(client, system_message, instructions)

    output = open(output_file, 'wb+')
    output.write(f'//{header}'.encode('utf-8'))
    output.write(response.encode('utf-8'))

def prompt_model(client, system_message, prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0]['message']['content']

if __name__ == '__main__':
    txt2pleco('transcript.txt', 'pleco_flashcards.txt', verbose=True)