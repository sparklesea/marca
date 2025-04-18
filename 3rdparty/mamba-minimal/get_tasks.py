import lm_eval
from lm_eval import evaluator, utils
from lm_eval.api.registry import ALL_TASKS, get_model
from lm_eval.logger import eval_logger, SPACING
import lm_eval.tasks
import os

import json
import collections

ROOT_DIR = "/root/huangshan/research/marca/3rdparty/mamba-minimal"

tasks_list = ["wikitext","lambada_openai","piqa","winogrande","arc_easy","hellaswag"]
# tasks_list = ["wikitext"]

task_names = utils.pattern_match(tasks_list, ALL_TASKS)
for task in [task for task in tasks_list if task not in task_names]:
    if os.path.isfile(task):
        config = utils.load_yaml_config(task)
        task_names.append(config)
task_missing = [task for task in tasks_list if task not in task_names]

if task_missing:
    missing = ", ".join(task_missing)
    eval_logger.error(
        f"Tasks were not found: {missing}\n"
        f"{SPACING}Try `lm-eval -h` for list of available tasks",
    )
    raise ValueError(
        f"Tasks {missing} were not found. Try `lm-eval -h` for list of available tasks."
    )


task_dict = lm_eval.tasks.get_task_dict(task_names)

# task_requests = []

for task_name, task in task_dict.items():
    task_requests = []
    task.build_all_requests(limit=None, rank=0, world_size=1)

    requests = collections.defaultdict(list)

    for instance in task.instances:
        reqtype = instance.request_type
        requests[reqtype].append(instance)

    for reqtype, instances in requests.items():
        if reqtype == "loglikelihood":
            task_requests.extend([instance.args[0] + instance.args[1] for instance in instances])
        elif reqtype == "loglikelihood_rolling":
            task_requests.extend([instance.args[0] for instance in instances])
        else:
            raise NotImplementedError(f"Request type {reqtype} not supported")

    with open(os.path.join(ROOT_DIR, f"lm_eval_benchmark/{task_name}.json"), "w") as f:
        json.dump(task_requests, f)
    
    print(task_name)
print(task_dict)
