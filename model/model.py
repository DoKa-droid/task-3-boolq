import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

def preprocessing_for_bert(data):
    # Создаем списки для хранения результатов
    input_ids = []
    attention_mask = []
    segment_ids = []

    questions = data.question.values
    passages = data.passage.values

    for question, passage in zip(questions, passages):
        encoded = tokenizer.encode_plus(question,  # Токенизируем вопрос и текст
                                        passage,
                                        # Определяем максимальную длину объединенного текста
                                        max_length=MAX_LEN,
                                        truncation=True,
                                        # Возвращаем тензор для выделения токенов для модели
                                        return_attention_mask=True)
        # Добавляем результат в массивы
        input_ids.append(encoded.get('input_ids'))
        attention_mask.append(encoded.get('attention_mask'))
        segment_ids.append(encoded.get('token_type_ids'))

    # Делаем из массивов тензоры
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    segment_ids = torch.tensor(segment_ids)

    return input_ids, attention_mask, segment_ids


my_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
my_model.load_state_dict(torch.load('my_model.pt', map_location=torch.device('cpu')))

# Загружаем BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
device = torch.device("cpu")
MAX_LEN = 512

if __name__ == '__main__':
    question = "is there a sequel to the movie the golden compass"
    passage = "The Golden Compass (film) -- In 2011, Philip Pullman remarked at the British Humanist Association annual conference that due to the first film's disappointing sales in the United States, there would not be any sequels made."
    d = {'question': [question], 'passage': [passage]}
    df = pd.DataFrame(data=d)

    # Предобрабатываем текст тренировочных данных
    test_inputs, test_masks, test_segment_ids = preprocessing_for_bert(df)

    # Загружаем части батчей в графический процессор
    input_ids = test_inputs.to(device)
    attention_mask = test_masks.to(device)
    segment_ids = test_segment_ids.to(device)

    outputs = my_model(input_ids, token_type_ids=segment_ids, attention_mask=attention_mask)

    probs = F.softmax(outputs[0], dim=1).detach().numpy()
    preds = probs[:, 1]
    y_pred = np.where(preds >= 0.5, 1, 0)
    answer = "No" if y_pred == 0 else "Yes"

    print(answer)
