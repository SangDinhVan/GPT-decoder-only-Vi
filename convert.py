import json

dataset = []
json_path = [f"data/data{i}.json" for i in range(1,8)]
json_path2 = [f"data/Translation/lima/data{i}.json" for i in range(98)]
def json_to_dataset(json_path, dataset):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        instruction = (item.get("instruction") or "").strip()
        input_text = (item.get("input") or "").strip()
        output = (item.get("output") or "").strip()

        prompt_clean = (f"{instruction} {input_text}".strip())
        output_clean = (output)

        prompt_clean = prompt_clean.lower()
        output_clean = output_clean.lower()
        full_line = f"<sos> USER {prompt_clean} AI {output_clean} <eos>\n"
        dataset.append(full_line)

def list_to_string(lst, sep=" "):
    if isinstance(lst, list):
        return sep.join([s.strip() for s in lst if isinstance(s, str)])
    return str(lst)

def json_to_dataset2(json_path, dataset):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        prompts = item.get("prompt", [])
        responses = item.get("response", [])

        prompt = list_to_string(prompts)
        response = list_to_string(responses)
        dataset.append(f"<sos> USER {prompt} AI {response} <eos>\n")




def load ():
    for path in json_path:
        json_to_dataset(path, dataset)

    # for path in json_path2:
    #     json_to_dataset2(path, dataset)
    return dataset
# load()
# dataset2 = []
# json_to_dataset2(json_path2[3], dataset2)
#
# print(dataset2[0])


