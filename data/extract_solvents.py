import json

path = 'dataset_close_5.json'

with open(path, 'r') as f:
    data = json.load(f)
sols = []
error = 0

for k, v in data.items():
    print(k, len(v[0]), '\n', v[0])
    if len(v[0]) == 3:
        for data in v:
            if len(data[1]) > 20:
                print(data)
                error += 1
            sols.append(data[1])

print('length before cleaning', len(sols))
sols = list(set(sols))
i = sols.index('gas')
sols.pop(i)
print('length after cleaning', len(sols))

with open('/mlx_devbox/users/howard.wang/playground/molllm/datasets/solvents_all.json', 'w+') as f:
    json.dump(sols, f)
with open('/mlx_devbox/users/howard.wang/playground/molllm/datasets/solvents_all.txt', 'w+') as f:
    for i in range(len(sols)):
        f.write(str(i).ljust(6, ' ') + sols[i] + '\n')
print(error)
