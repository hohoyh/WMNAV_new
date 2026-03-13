import logging
from openai import OpenAI
import base64
import numpy as np
from PIL import Image
import io
import os

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def encode_image(image):
    try:
        # 将numpy数组转换回PIL图像
        image = Image.fromarray(image[:, :, :3], mode='RGB')

        # 将图像保存到字节流中
        buffered = io.BytesIO()
        # 🔴 核心修复：改为 JPEG 格式，并设置 quality=80 压缩画质。体积将缩小近 10 倍！
        image.save(buffered, format="JPEG", quality=80) 

        # 将字节流编码为Base64
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to convert image to base64: {e}")
# def encode_image(image):
#     try:
#         # 将numpy数组转换回PIL图像
#         image = Image.fromarray(image[:, :, :3], mode='RGB')

#         # 将图像保存到字节流中
#         buffered = io.BytesIO()
#         image.save(buffered, format="PNG")

#         # 将字节流编码为Base64
#         return base64.b64encode(buffered.getvalue()).decode('utf-8')
#     except Exception as e:
#         raise RuntimeError(f"Failed to convert image to base64: {e}")


# class GeminiVLM:
#     """
#     A specific implementation of a VLM using the Gemini API for image and text inference.
#     """

#     def __init__(self, model="gemini-2.0-flash", system_instruction=None):
#         """
#         Initialize the Gemini model with specified configuration.

#         Parameters
#         ----------
#         model : str
#             The model version to be used.
#         system_instruction : str, optional
#             System instructions for model behavior.
#         """
#         self.name = model
#         self.client = OpenAI(
#             api_key=os.environ.get("GEMINI_API_KEY"),
#             base_url=os.environ.get("GEMINI_BASE_URL")
#         )

#         self.system_instruction = system_instruction

#         self.spend = 0
#         if '1.5-flash' in self.name:
#             self.cost_per_input_token = 0.075 / 1_000_000
#             self.cost_per_output_token = 0.3 / 1_000_000
#         elif '1.5-pro' in self.name:
#             self.cost_per_input_token = 1.25 / 1_000_000
#             self.cost_per_output_token = 5 / 1_000_000
#         else:
#             self.cost_per_input_token = 0.1 / 1_000_000
#             self.cost_per_output_token = 0.4 / 1_000_000

#     def call_chat(self, image: list[np.array], text_prompt: str):
#         base64_image = encode_image(image[0])
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.name,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": self.system_instruction
#                     },
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": text_prompt},
#                             {
#                                 "type": "image_url",
#                                 "image_url": {
#                                     "url": f"data:image/png;base64,{base64_image}"
#                                 },
#                             },
#                         ],
#                     }
#                 ],
#                 max_tokens=500,
#                 temperature=0,
#                 top_p=1,
#                 stream=False
#             )
            
#             # 1. 先算钱！(确保此时 response 还是原生对象)
#             if hasattr(response, 'usage') and response.usage is not None:
#                 self.spend += (response.usage.prompt_tokens * self.cost_per_input_token +
#                                response.usage.completion_tokens * self.cost_per_output_token)
#             else:
#                 # 兼容部分不返回 usage 的中转节点
#                 self.spend += 0 
                
#             # 2. 再脱壳！(赋值给新变量，千万别覆盖 response)
#             raw_text = response.choices[0].message.content
#             clean_text = raw_text.replace("```json", "").replace("```", "").strip()
#             return clean_text

#         except Exception as e:
#             error_msg = str(e)
#             print(f"⚠️ 真实的 API 报错信息: {error_msg}")
#             return f"API ERROR: {error_msg}"

#         # # 🔴 过滤掉 GPT 喜欢带的 markdown 标记，确保返回纯净的字典字符串
#         # raw_text = response.choices[0].message.content
#         # clean_text = raw_text.replace("```json", "").replace("```", "").strip()
#         # return clean_text
    
#         # return response.choices[0].message.content

#     # def call(self, image: list[np.array], text_prompt: str):
#     #     base64_image = encode_image(image[0])
#     #     try:
#     #         response = self.client.chat.completions.create(
#     #             model=self.name,
#     #             messages=[
#     #                 {
#     #                     "role": "user",
#     #                     "content": [
#     #                         {"type": "text", "text": text_prompt},
#     #                         {
#     #                             "type": "image_url",
#     #                             "image_url": {
#     #                                 "url": f"data:image/png;base64,{base64_image}"
#     #                             },
#     #                         },
#     #                     ],
#     #                 }
#     #             ],
#     #             max_tokens=500,
#     #             temperature=0,
#     #             top_p=1,
#     #             stream=False  # 是否开启流式输出
#     #         )
#     #         self.spend += (response.usage.prompt_tokens * self.cost_per_input_token +
#     #                        response.usage.completion_tokens * self.cost_per_output_token)
#     #     except Exception as e:
#     #         print(f"GEMINI API ERROR: {e}")
#     #         return "GEMINI API ERROR"
#     #     return response.choices[0].message.content
#     def call(self, image: list[np.array], text_prompt: str):
#         base64_image = encode_image(image[0])
#         try:
#             messages = []
#             if self.system_instruction:
#                 messages.append({"role": "system", "content": self.system_instruction})
                
#             messages.append({
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": text_prompt},
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/png;base64,{base64_image}"
#                         }
#                     },
#                 ],
#             })

#             response = self.client.chat.completions.create(
#                 model=self.name,
#                 messages=messages,
#                 max_tokens=500,
#                 temperature=0,
#                 top_p=1,
#                 stream=False
#             )
            
#             # 1. 先算钱
#             if hasattr(response, 'usage') and response.usage is not None:
#                 self.spend += (response.usage.prompt_tokens * self.cost_per_input_token +
#                                response.usage.completion_tokens * self.cost_per_output_token)
#             else:
#                 self.spend += 0
                
#             # 2. 再脱壳
#             raw_text = response.choices[0].message.content
#             clean_text = raw_text.replace("```json", "").replace("```", "").strip()
#             return clean_text

#         except Exception as e:
#             error_msg = str(e)
#             print(f"⚠️ 真实的 API 报错信息: {error_msg}")
#             return f"API ERROR: {error_msg}"

#     def reset(self):
#         """
#         Reset the context state of the VLM agent.
#         """
#         pass

#     def get_spend(self):
#         """
#         Retrieve the total spend on model usage.
#         """
#         return self.spend

class GeminiVLM:
    """
    A specific implementation of a VLM using the Gemini API for image and text inference.
    """

    def __init__(self, model="gemini-2.0-flash", system_instruction=None):
        self.name = model
        self.client = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url=os.environ.get("GEMINI_BASE_URL")
        )

        self.system_instruction = system_instruction

        self.spend = 0
        if '1.5-flash' in self.name:
            self.cost_per_input_token = 0.075 / 1_000_000
            self.cost_per_output_token = 0.3 / 1_000_000
        elif '1.5-pro' in self.name:
            self.cost_per_input_token = 1.25 / 1_000_000
            self.cost_per_output_token = 5 / 1_000_000
        else:
            self.cost_per_input_token = 0.1 / 1_000_000
            self.cost_per_output_token = 0.4 / 1_000_000

    def call_chat(self, image: list[np.array], text_prompt: str, max_tokens: int = 500):
        import time
        base64_image = encode_image(image[0])
        
        # 🔴 给云端大脑加上最多 8 次的抗拥堵重试机制
        for attempt in range(8):
            try:
                response = self.client.chat.completions.create(
                    model=self.name,
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_instruction
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                },
                            ],
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=0,
                    top_p=1,
                    stream=False
                )
                
                # 🔴 照妖镜：拦截中转代理瞎返回的纯文本报错（直接触发异常去休眠重试）
                if isinstance(response, str):
                    raise Exception(f"Proxy Alert (Plain Text): {response}")
                elif isinstance(response, dict):
                    # 兼容极少数把字典当作裸对象返回的代理
                    raw_text = response.get('choices', [{}])[0].get('message', {}).get('content', str(response))
                    return raw_text.replace("```json", "").replace("```", "").strip()
                
                # 1. 先算钱
                if hasattr(response, 'usage') and response.usage is not None:
                    self.spend += (response.usage.prompt_tokens * self.cost_per_input_token +
                                   response.usage.completion_tokens * self.cost_per_output_token)
                else:
                    self.spend += 0 
                    
                # 2. 再脱壳
                raw_text = response.choices[0].message.content
                clean_text = raw_text.replace("```json", "").replace("```", "").strip()
                
                # 🔴 主动降速：成功后睡 1.5 秒，防连续射击被封
                time.sleep(1.5)
                return clean_text

            except Exception as e:
                # 🔴 指数退避重试：失败就睡一会再试
                wait_time = (attempt + 1) * 4
                print(f"⚠️ 云端 API 拥堵/报错 (第 {attempt + 1} 次重试)，主动休眠 {wait_time} 秒... [原因: {str(e)}]")
                time.sleep(wait_time)
                
        return "API ERROR: 重试 8 次均失败"

    def call(self, image: list[np.array], text_prompt: str):
        import time
        base64_image = encode_image(image[0])
        
        for attempt in range(8):
            try:
                messages = []
                if self.system_instruction:
                    messages.append({"role": "system", "content": self.system_instruction})
                    
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                    ],
                })

                response = self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    max_tokens=500,
                    temperature=0,
                    top_p=1,
                    stream=False
                )
                
                # 🔴 照妖镜
                if isinstance(response, str):
                    raise Exception(f"Proxy Alert (Plain Text): {response}")
                elif isinstance(response, dict):
                    raw_text = response.get('choices', [{}])[0].get('message', {}).get('content', str(response))
                    return raw_text.replace("```json", "").replace("```", "").strip()
                
                if hasattr(response, 'usage') and response.usage is not None:
                    self.spend += (response.usage.prompt_tokens * self.cost_per_input_token +
                                   response.usage.completion_tokens * self.cost_per_output_token)
                else:
                    self.spend += 0
                    
                raw_text = response.choices[0].message.content
                clean_text = raw_text.replace("```json", "").replace("```", "").strip()
                
                time.sleep(1.5)
                return clean_text

            except Exception as e:
                wait_time = (attempt + 1) * 4
                print(f"⚠️ 云端 API 拥堵/报错 (第 {attempt + 1} 次重试)，主动休眠 {wait_time} 秒... [原因: {str(e)}]")
                time.sleep(wait_time)
                
        return "API ERROR: 重试 8 次均失败"

    def reset(self):
        pass

    def get_spend(self):
        return self.spend


# class QwenVLM:
#     """
#     A specific implementation of a VLM using the Qwen API for image and text inference.
#     """

#     def __init__(self, model="Qwen/Qwen2.5-VL-7B-Instruct", system_instruction=None):
#         """
#         Initialize the Qwen model with specified configuration.

#         Parameters
#         ----------
#         model : str
#             The model version to be used.
#         system_instruction : str, optional
#             System instructions for model behavior.
#         """
#         # self.name = model
#         # self.client = OpenAI(
#         #     api_key=os.environ.get("GEMINI_API_KEY"),
#         #     base_url=os.environ.get("GEMINI_BASE_URL")
#         # )

#         # self.system_instruction = system_instruction

#         # self.spend = 0
#         self.name = model
        
#         # 🔴 核心修改：让 Qwen 强制连接你电脑本地的 vLLM 服务！
#         self.client = OpenAI(
#             api_key="EMPTY",  # 本地服务不需要真实的秘钥
#             base_url="http://localhost:8000/v1" # 默认的 vLLM 端口
#         )

#         self.system_instruction = system_instruction
#         self.spend = 0

#     def call_chat(self, image: list[np.array], text_prompt: str, max_tokens: int = 500):
#             import time
#             base64_image = encode_image(image[0])
            
#             # 允许最多 8 次重试，给足恢复时间
#             for attempt in range(8):
#                 try:
#                     messages = []
#                     if self.system_instruction:
#                         messages.append({"role": "system", "content": self.system_instruction})
                        
#                     messages.append({
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": text_prompt},
#                             {
#                                 "type": "image_url",
#                                 "image_url": {
#                                     "url": f"data:image/png;base64,{base64_image}"
#                                 }
#                             },
#                         ],
#                     })

#                     response = self.client.chat.completions.create(
#                         model=self.name,
#                         messages=messages,
#                         max_tokens=max_tokens,
#                         temperature=0,
#                         top_p=1,
#                         stream=False
#                     )
                    
#                     # 🔴 核心策略 1：如果返回异常文本，主动抛出异常，强制进入下方的 except 重试环节！
#                     if isinstance(response, str):
#                         raise Exception(f"Proxy Alert: {response}")
#                     elif isinstance(response, dict):
#                         raw_text = response.get('choices', [{}])[0].get('message', {}).get('content', str(response))
#                         return raw_text.replace("```json", "").replace("```", "").strip()
                    
#                     # 正常算钱与脱壳
#                     if hasattr(response, 'usage') and response.usage is not None:
#                         self.spend += (response.usage.prompt_tokens * self.cost_per_input_token +
#                                     response.usage.completion_tokens * self.cost_per_output_token)
                        
#                     raw_text = response.choices[0].message.content
#                     clean_text = raw_text.replace("```json", "").replace("```", "").strip()
                    
#                     # 🔴 核心策略 2：主动降速。每次成功后强制睡 1.5 秒，避免连续射击触发封控
#                     time.sleep(1.5)
#                     return clean_text

#                 except Exception as e:
#                     # 🔴 核心策略 3：指数退避重试 (Exponential Backoff)
#                     # 第一次失败睡 4 秒，第二次 8 秒，第三次 12 秒... 越往后等越久，直到平台配额恢复
#                     wait_time = (attempt + 1) * 4  
#                     print(f"⚠️ API 拥堵/报错 (第 {attempt + 1} 次重试)，主动休眠 {wait_time} 秒... [原因: {str(e)}]")
#                     time.sleep(wait_time)
                    
#             return "API ERROR: 重试 8 次均失败"

#     def call(self, image: list[np.array], text_prompt: str):
#         import time
#         base64_image = encode_image(image[0])
        
#         for attempt in range(8):
#             try:
#                 messages = []
#                 if self.system_instruction:
#                     messages.append({"role": "system", "content": self.system_instruction})
                    
#                 messages.append({
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": text_prompt},
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/png;base64,{base64_image}"
#                             }
#                         },
#                     ],
#                 })

#                 response = self.client.chat.completions.create(
#                     model=self.name,
#                     messages=messages,
#                     max_tokens=500,
#                     temperature=0,
#                     top_p=1,
#                     stream=False
#                 )
                
#                 if isinstance(response, str):
#                     raise Exception(f"Proxy Alert: {response}")
#                 elif isinstance(response, dict):
#                     raw_text = response.get('choices', [{}])[0].get('message', {}).get('content', str(response))
#                     return raw_text.replace("```json", "").replace("```", "").strip()
                
#                 if hasattr(response, 'usage') and response.usage is not None:
#                     self.spend += (response.usage.prompt_tokens * self.cost_per_input_token +
#                                    response.usage.completion_tokens * self.cost_per_output_token)
                    
#                 raw_text = response.choices[0].message.content
#                 clean_text = raw_text.replace("```json", "").replace("```", "").strip()
                
#                 time.sleep(1.5)
#                 return clean_text

#             except Exception as e:
#                 wait_time = (attempt + 1) * 4
#                 print(f"⚠️ API 拥堵/报错 (第 {attempt + 1} 次重试)，主动休眠 {wait_time} 秒... [原因: {str(e)}]")
#                 time.sleep(wait_time)
                
#         return "API ERROR: 重试 8 次均失败"
#     def reset(self):
#         """
#         Reset the context state of the VLM agent.
#         """
#         pass

#     def get_spend(self):
#         """
#         Retrieve the total spend on model usage.
#         """
#         return self.spend


class QwenVLM:
    """
    A specific implementation of a VLM using the Qwen API for image and text inference.
    """
    def __init__(self, model="Qwen2.5-VL-3B-Instruct", system_instruction=None):
        self.name = model
        self.client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )
        self.system_instruction = system_instruction
        self.spend = 0

    def call_chat(self, image: list[np.array], text_prompt: str, max_tokens: int = 500):
        import time
        base64_image = encode_image(image[0])
        
        for attempt in range(3): # 本地模型失败重试3次足够了
            try:
                messages = []
                if self.system_instruction:
                    messages.append({"role": "system", "content": self.system_instruction})
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ],
                })

                response = self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0,
                    top_p=1,
                    stream=False
                )
                
                # 🔴 核心修复：本地免费，不算钱了！直接读取返回文本！
                raw_text = response.choices[0].message.content
                clean_text = raw_text.replace("```json", "").replace("```", "").strip()
                
                return clean_text # 本地没有并发限制，直接返回，不需要 sleep！

            except Exception as e:
                print(f"⚠️ 本地 API 报错 (重试 {attempt + 1}): {e}")
                time.sleep(2)
                
        return "API ERROR"

    def call(self, image: list[np.array], text_prompt: str):
        import time
        base64_image = encode_image(image[0])
        
        for attempt in range(3):
            try:
                messages = []
                if self.system_instruction:
                    messages.append({"role": "system", "content": self.system_instruction})
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ],
                })

                response = self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    max_tokens=500,
                    temperature=0,
                    top_p=1,
                    stream=False
                )
                
                # 🔴 核心修复：本地免费，不算钱了！
                raw_text = response.choices[0].message.content
                clean_text = raw_text.replace("```json", "").replace("```", "").strip()
                
                return clean_text 

            except Exception as e:
                print(f"⚠️ 本地 API 报错 (重试 {attempt + 1}): {e}")
                time.sleep(2)
                
        return "API ERROR"

    def reset(self):
        pass

    def get_spend(self):
        return self.spend