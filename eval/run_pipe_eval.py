from stage_ea_model import StageEaModel
from stage_ea_config import StageEaConfig
from fastchat.model import get_conversation_template
from fastchat.llm_judge.common import load_questions
# from datetime import timedelta
from pipeline_utils import calculate_model_size_with_buffers, get_time_str
import torch
import torch.distributed as dist
from tqdm import tqdm
# import time
import argparse
import os
import warnings
# from torch.profiler import ProfilerActivity
from profiler.profiler import prof, is_strictly_ascending, save_as
from contextlib import nullcontext
from config.run_config import config as run_config
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated.*")
import time
import random
import numpy as np

rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])

# torch.manual_seed(1234)
# os.environ['TORCH_DISTRIBUTED_DEFAULT_TIMEOUT'] = '120'
    
def run_eval(args):
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # device = rank % torch.cuda.device_count()
    device = 0
    torch.cuda.set_device(device)
    print(f'rank={rank}, world_size={world_size}, device={device}')
    
    # with prof.profile_context(f"Rank {rank}: loading stage model", device=f"cuda:{device}"):
    print(f'Load model from {run_config.base_model_dir}...')
    print(f'Load EAGLE model from {run_config.EAGLE_model_path}...')
    stage_model = StageEaModel.from_pretrained(
        stage_base_model_path=run_config.base_model_dir + f"/stage_model_{rank}",
        ea_model_path=run_config.EAGLE_model_path if rank == 0 else None,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # max_memory={"cpu": "1GB"},
        use_safetensors=True,
        quantization_config=run_config.quant_config,
        device_map=f"cuda:{device}",
        total_token=run_config.init_total_token,
        depth=run_config.init_depth,
        top_k=run_config.init_topk #if run_config.pipeline_type != "pipedec" else run_config.init_topk_pipedec,
    )
    
    stage_model.eval()
    stage_model.stage_base_model.config.max_position_embeddings = 2560
    print(f'Rank {rank}: max_position_embeddings={stage_model.stage_base_model.config.max_position_embeddings}')

    # assert run_config.pipeline_type in ["naive", "pruned", "continuous", "pipedec"]
    # assert run_config.mode in ["eval", "demo"]
    
    questions = load_questions(run_config.question_paths[0], run_config.question_begin, run_config.question_end)
    
    ###########################################
    #warmup
    ###########################################
    if run_config.warmup:
        q = questions[0]
        cnt = tqdm(range(run_config.warmup_repeat), desc="Warmup") if rank == 0 else range(run_config.warmup_repeat)
        for _ in cnt:
            torch.manual_seed(0) 
                
            if rank == 0:
                if "llama2" in args.model_name:
                    conv = get_conversation_template("llama-2-chat")
                    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
                    conv.system_message = sys_p
                elif "vicuna" in args.model_name:
                    conv = get_conversation_template("vicuna")
                elif "llama3" in args.model_name:
                    messages = [
                        {"role": "system",
                        "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
                    ]

            for k in range(len(q["turns"])):
                if rank == 0:
                    q_turn = q["turns"][k]
                    
                    if "llama3" in args.model_name:
                        messages.append({
                            "role": "user",
                            "content": q_turn
                        })
                        prompt = stage_model.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        input_ids = stage_model.tokenizer([prompt],add_special_tokens=False,).input_ids
                    else:
                        conv.append_message(conv.roles[0], q_turn)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt() #
                        if "llama2" in args.model_name:
                            prompt = prompt + " "
                        input_ids = stage_model.tokenizer([prompt]).input_ids
                        
                    input_ids = torch.as_tensor(input_ids).cuda()
                
                outputs = run(
                    stage_model, 
                    input_ids if rank == 0 else None, 
                    run_config.temperatures[0],
                    run_config.pipeline_types[0],
                    run_config.log if rank == 0 else False, 
                    None
                )
                
                if rank == 0:  # only for greedy decoding test!!!
                    if run_config.log:
                        output_ids, new_tokens, idx, turns = outputs
                    else:
                        output_ids = outputs
                        
                    output_ids = output_ids[0][len(input_ids[0]):]
                    
                    if "llama3" in args.model_name:
                        stop_token_ids = [
                            stage_model.tokenizer.eos_token_id,
                            stage_model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        ]
                        if stop_token_ids:
                            stop_token_ids_index = [
                                i
                                for i, id in enumerate(output_ids)
                                if id in stop_token_ids
                            ]
                            if len(stop_token_ids_index) > 0:
                                output_ids = output_ids[: stop_token_ids_index[0]]
                    else:
                        if conv.stop_token_ids:
                            stop_token_ids_index = [
                                i
                                for i, id in enumerate(output_ids)
                                if id in conv.stop_token_ids
                            ]
                            if len(stop_token_ids_index) > 0:
                                output_ids = output_ids[: stop_token_ids_index[0]]

                    output = stage_model.tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    
                    if "llama3" in args.model_name:
                        for special_token in stage_model.tokenizer.special_tokens_map.values():
                            if isinstance(special_token, list):
                                for special_tok in special_token:
                                    output = output.replace(special_tok, "")
                            else:
                                output = output.replace(special_token, "")
                        output = output.strip()
                        
                        messages.append({
                            "role": "assistant",
                            "content": output
                        })
                    else:
                        conv.stop_str = "</s>"
                        if conv.stop_str and output.find(conv.stop_str) > 0:
                            output = output[: output.find(conv.stop_str)]
                        for special_token in stage_model.tokenizer.special_tokens_map.values():
                            if isinstance(special_token, list):
                                for special_tok in special_token:
                                    output = output.replace(special_tok, "")
                            else:
                                output = output.replace(special_token, "")
                        output = output.strip()

                        if conv.name == "xgen" and output.startswith("Assistant:"):
                            output = output.replace("Assistant:", "", 1).strip()
                            
                        conv.messages[-1][-1] = output
        
    ###########################################
    #test
    ###########################################
    record_path = f"{run_config.model_name}-{args.extra_name}.txt"
    for temperature in run_config.temperatures:
        for pipeline_type in run_config.pipeline_types:
            for _ in range(run_config.error_repeat):
                for question_path in run_config.question_paths:
                    questions = load_questions(question_path, run_config.question_begin, run_config.question_end)
                
                    cnt = tqdm(range(len(questions)), desc=question_path) if rank == 0 else range(len(questions))
                    new_tokens_list = []
                    wall_time_list = []
                    idx_list = []
                    turns_list = []
                    
                    for i in cnt:
                        q = questions[i]
                        
                        for j in range(run_config.test_repeat): 
                            torch.manual_seed(j)
                            # random.seed(j) #
                            # np.random.seed(j) #
                            
                            if rank == 0:
                                if "llama2" in args.model_name:
                                    conv = get_conversation_template("llama-2-chat")
                                    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
                                    conv.system_message = sys_p
                                    # conv.stop_token_ids = [2] #
                                    
                                elif "vicuna" in args.model_name:
                                    conv = get_conversation_template("vicuna")
                                    # conv.stop_token_ids = [2] #
                                    
                                elif "llama3" in args.model_name:
                                    messages = [
                                        {"role": "system",
                                        "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
                                    ]
                            
                            for k in range(len(q["turns"])):
                                if rank == 0:
                                    q_turn = q["turns"][k]
                                    if "llama3" in args.model_name:
                                        messages.append({
                                            "role": "user",
                                            "content": q_turn
                                        })
                                        prompt = stage_model.tokenizer.apply_chat_template(
                                            messages,
                                            tokenize=False,
                                            add_generation_prompt=True,
                                        )
                                        input_ids = stage_model.tokenizer([prompt],add_special_tokens=False,).input_ids
                                    else:
                                        conv.append_message(conv.roles[0], q_turn)
                                        conv.append_message(conv.roles[1], None)
                                        prompt = conv.get_prompt() #
                                        if "llama2" in args.model_name:
                                            prompt = prompt + " "
                                        input_ids = stage_model.tokenizer([prompt]).input_ids
                                        
                                    input_ids = torch.as_tensor(input_ids).cuda()
                                
                                with prof.profile_context(f"Rank {rank}: {pipeline_type} pipeline", device=f"cuda:{device}") if run_config.prof else nullcontext():
                                    start_time = time.time()
                                    outputs = run(
                                        stage_model, 
                                        input_ids if rank == 0 else None, 
                                        temperature,
                                        pipeline_type,
                                        run_config.log if rank == 0 else False, 
                                        prof if run_config.prof else None
                                    )
                                    torch.cuda.synchronize()
                                    end_time = time.time()
                                    wall_time = end_time - start_time
                                
                                if rank == 0:  # only for greedy decoding test!!!
                                    if run_config.log:
                                        output_ids, new_tokens, idx, turns = outputs
                                    else:
                                        output_ids = outputs
                                    
                                    output_ids = output_ids[0][len(input_ids[0]):]
                                    if "llama3" in args.model_name:
                                        stop_token_ids = [
                                            stage_model.tokenizer.eos_token_id,
                                            stage_model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                                        ]
                                        if stop_token_ids:
                                            stop_token_ids_index = [
                                                i
                                                for i, id in enumerate(output_ids)
                                                if id in stop_token_ids
                                            ]
                                            if len(stop_token_ids_index) > 0:
                                                output_ids = output_ids[: stop_token_ids_index[0]]
                                    else:
                                        
                                        if conv.stop_token_ids:
                                            stop_token_ids_index = [
                                                i
                                                for i, id in enumerate(output_ids)
                                                if id in conv.stop_token_ids
                                            ]
                                            if len(stop_token_ids_index) > 0:
                                                output_ids = output_ids[: stop_token_ids_index[0]]

                                    output = stage_model.tokenizer.decode(
                                        output_ids,
                                        spaces_between_special_tokens=False,
                                    )
                                    
                                    if "llama3" in args.model_name:
                                        for special_token in stage_model.tokenizer.special_tokens_map.values():
                                            if isinstance(special_token, list):
                                                for special_tok in special_token:
                                                    output = output.replace(special_tok, "")
                                            else:
                                                output = output.replace(special_token, "")
                                        output = output.strip()
                                        
                                        messages.append({
                                            "role": "assistant",
                                            "content": output
                                        })
                                    else:
                                            # conv.stop_str = "</s>"
                                        if conv.stop_str and output.find(conv.stop_str) > 0:
                                            output = output[: output.find(conv.stop_str)]
                                        for special_token in stage_model.tokenizer.special_tokens_map.values():
                                            if isinstance(special_token, list):
                                                for special_tok in special_token:
                                                    output = output.replace(special_tok, "")
                                            else:
                                                output = output.replace(special_token, "")
                                        output = output.strip()

                                        if conv.name == "xgen" and output.startswith("Assistant:"):
                                            output = output.replace("Assistant:", "", 1).strip()
                                            
                                        conv.messages[-1][-1] = output
                                
                                if rank == 0:
                                    new_tokens_list.append(output_ids.shape[0])
                                    wall_time_list.append(wall_time)
                                    if run_config.log:
                                        idx_list.append(idx)
                                        turns_list.append(turns)
                            
                    dist.barrier()
                    # if rank == 0 or rank == world_size - 1:
                    if run_config.prof:
                        prof.print_all_events()
                    
                    if rank == 0:
                        throughput = sum(new_tokens_list) / sum(wall_time_list)
                        avg_latency = sum(wall_time_list) / len(wall_time_list)
                        print(f'temperature: {temperature}, pipeline_type: {pipeline_type}, question_path: {question_path}, question_begin: {run_config.question_begin}, question_end: {run_config.question_end}')
                        print(f'throughput: {throughput}, avg_latency: {avg_latency}')
                        if run_config.log:
                            total_rounds = sum(idx_list)
                            total_turns = sum(turns_list)
                            print(f'rounds: {sum(idx_list)}, new_tokens: {sum(new_tokens_list)}, avg_accept_length: {sum(new_tokens_list)/sum(idx_list)}')
                            print(f'turns: {sum(turns_list)}, new_tokens: {sum(new_tokens_list)}, avg_accept_length: {sum(new_tokens_list)/sum(turns_list)}')
                        if run_config.eval_record:
                            with open(record_path, 'a') as f:
                                f.write(f'temperature: {temperature}, pipeline_type: {pipeline_type}, question_path: {question_path}, question_begin: {run_config.question_begin}, question_end: {run_config.question_end}\n')
                                f.write(f'new_tokens_list: {new_tokens_list}\n')
                                f.write(f'wall_time_list: {wall_time_list}\n')
                                f.write(f'throughput: {throughput}\n')
                                f.write(f'avg_latency: {avg_latency}\n')
                                if run_config.log:
                                    f.write(f'avg_accept_length: {sum(new_tokens_list)/sum(idx_list)}\n')
                                    f.write(f'turns: {sum(turns_list)}, new_tokens: {sum(new_tokens_list)}, avg_accept_length: {sum(new_tokens_list)/sum(turns_list)}\n')
                                f.write(f'---------------------------------------------------------------------------------------------------------\n')
                    dist.barrier()
                    
                # dist.barrier()
    if hasattr(stage_model, "comm"):
        stage_model.comm.stop()
    dist.destroy_process_group()
    
    # reset traffic
    if run_config.hardware == "jetson" and run_config.set_network:
        stage_model.comm.reset_traffic()
    

def run(stage_model, input_ids, temperature, pipeline_type, log=False, profiler=None):
    outputs = stage_model.stage_generate(
        input_ids=input_ids,
        temperature=temperature,
        max_new_tokens=run_config.max_new_tokens,
        log=log,
        pipeline_type=pipeline_type,
        profiler=profiler,
    )
    if dist.get_rank() == 0:
        return outputs
            
if __name__ == "__main__":
    assert run_config.mode == "eval"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=run_config.model_name)
    parser.add_argument("--base_model_dir", type=str, default=run_config.base_model_dir)
    parser.add_argument("--EAGLE_model_path", type=str, default=run_config.EAGLE_model_path)
    parser.add_argument("--extra_name", type=str, default="")
    args = parser.parse_args()
    run_eval(args)
