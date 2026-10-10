import re
import torch
from typing import List, Dict

class KeyObjectTokenTracker:
    def __init__(self, tokenizer_wrapper):
        """
        tokenizer_wrapper: DiffSynth 的 HuggingfaceTokenizer 实例
        """
        self.wrapper = tokenizer_wrapper
        # 拿到真正的 HuggingFace Tokenizer (Fast版本)
        self.hf_tokenizer = tokenizer_wrapper.tokenizer

        if not self.hf_tokenizer.is_fast:
            raise RuntimeError("UMT5 必须使用 Fast 版本才能追踪 Token 位置")

    def find_key_object_indices(
        self,
        prompt: str,
        key_objects: List[str],
        case_sensitive: bool = False,
        whole_word: bool = True,
        verbose: bool = True
    ) -> Dict[str, List[int]]:
        """
        查找关键词在 tokenized prompt 中的位置

        Args:
            prompt: 输入文本
            key_objects: 关键词列表
            case_sensitive: 是否大小写敏感（默认 False）
            whole_word: 是否只匹配完整单词（默认 True，避免 "cat" 匹配到 "concatenate"）
            verbose: 是否打印调试信息（默认 True）

        Returns:
            Dict[str, List[int]]: 关键词到 token 索引列表的映射
        """
        # 1. 必须执行与包装类完全相同的清洗逻辑
        # 如果不清洗，字符偏移量(char offsets)会对不上
        cleaned_prompt = self.wrapper._clean(prompt) if self.wrapper.clean else prompt

        # 2. 直接调用底层的 hf_tokenizer，确保返回的是 BatchEncoding 对象
        encoding = self.hf_tokenizer(
            cleaned_prompt,
            add_special_tokens=True,
            return_offsets_mapping=True,
            padding='max_length',           # 必须与 Wan 训练/推理时的长度一致
            truncation=True,
            max_length=self.wrapper.seq_len or 512,
            return_tensors="pt"             # 这里返回的是 BatchEncoding
        )

        # 获取 token 列表用于 Debug
        input_ids = encoding["input_ids"][0]
        tokens = self.hf_tokenizer.convert_ids_to_tokens(input_ids)

        object_indices = {}
        for obj in key_objects:
            # 同样对关键词进行清洗（如果关键词也是 prompt 的一部分）
            cleaned_obj = self.wrapper._clean(obj) if self.wrapper.clean else obj

            # 构建正则表达式
            pattern = re.escape(cleaned_obj)
            if whole_word:
                # 添加词边界，避免 "cat" 匹配到 "concatenate"
                pattern = rf"\b{pattern}\b"

            # 设置大小写敏感标志
            flags = 0 if case_sensitive else re.IGNORECASE

            indices = set()
            match_count = 0

            # 在清洗后的 prompt 中查找
            for match in re.finditer(pattern, cleaned_prompt, flags):
                start_char, end_char = match.span()

                # 使用 char_to_token 映射 (针对 batch index 0)
                start_token = encoding.char_to_token(0, start_char)
                end_token = encoding.char_to_token(0, end_char - 1)

                if start_token is not None and end_token is not None:
                    for i in range(start_token, end_token + 1):
                        indices.add(i)
                    match_count += 1
                else:
                    if verbose:
                        print(
                            f"[TokenTracker] 警告: 关键词 '{obj}' 在字符位置 ({start_char}, {end_char}) "
                            f"无法映射到 token (可能在特殊 token 上)"
                        )

            object_indices[obj] = sorted(list(indices))

            # 验证打印
            if verbose:
                matched_tokens = [tokens[i] for i in object_indices[obj]]
                print(
                    f"[TokenTracker] '{obj}' -> "
                    f"找到 {match_count} 处匹配, "
                    f"覆盖 {len(object_indices[obj])} 个 tokens: "
                    f"{matched_tokens} (indices: {object_indices[obj]})"
                )

        return object_indices