from simple_test.GPT2_autoregressive_inference import *
from transformers import PretrainedConfig


config, tokenizer, model = load_pretrained_local(model_path)

model.eval()

# # 插入量化和反量化操作
# model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
#
# # 准备量化模型
# model_prepared = quan.prepare(model)
#
# # 校准模型（使用校准数据集）
# calibration_data = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]
# model_prepared(calibration_data)
#
# # 完成量化
# model_quantized = quan.convert(model_prepared)
model = model.half()

# 保存量化模型的权重
quantized_model_path = os.path.join(model_path, 'float16_model.bin')
torch.save(model.state_dict(), quantized_model_path)


class HalfGPT2Config(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# class HalfPrecisionGPT2Config(PretrainedConfig):
#     model_type = "half_precision_gpt2"
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)


class HalfGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config: HalfGPT2Config):
        super(HalfGPT2LMHeadModel, self).__init__(config)
        self.config_class = HalfGPT2Config
        self = self.half()




# 注册自定义模型类
from transformers import AutoModel

AutoModel.register(HalfGPT2LMHeadModel.config_class, HalfGPT2LMHeadModel)

# 加载量化后的模型
model_quantized_loaded = HalfGPT2LMHeadModel.from_pretrained(quantized_model_path, from_pt=True)

# 测试加载的模型
input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]
outputs = model_quantized_loaded(input_ids)
print(outputs.last_hidden_state)
