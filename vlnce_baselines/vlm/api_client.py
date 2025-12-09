"""
API配置和客户端基类
==================
统一管理API配置和调用逻辑
"""
import os
import yaml
import base64
import json
import re
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class APIConfig:
    """统一API配置类"""
    
    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 验证必要字段
        required = ['api_key', 'base_url', 'model']
        missing = [f for f in required if f not in self.config or not self.config[f]]
        if missing:
            raise ValueError(f"配置文件缺少必要字段: {', '.join(missing)}")
    
    @property
    def api_key(self) -> str:
        return self.config['api_key']
    
    @property
    def base_url(self) -> str:
        return self.config['base_url']
    
    @property
    def model(self) -> str:
        return self.config['model']
    
    @property
    def temperature(self) -> float:
        return self.config.get('temperature', 0.1)
    
    @property
    def max_tokens(self) -> int:
        return self.config.get('max_tokens', 2000)
    
    @property
    def timeout(self) -> int:
        return self.config.get('timeout', 60)
    
    def get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }


class BaseAPIClient(ABC):
    """API客户端基类"""
    
    def __init__(self, config: APIConfig):
        self.config = config
    
    @staticmethod
    def encode_image_base64(image_path: str) -> str:
        """编码图像为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    @staticmethod
    def clean_json_response(text: str) -> str:
        """清理响应文本"""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        return text.strip()
    
    @staticmethod
    def extract_json_object(text: str) -> Optional[str]:
        """提取JSON对象"""
        match = re.search(r'\{[\s\S]*\}', text)
        return match.group(0) if match else None
    
    def parse_json_response(self, response_text: str) -> Optional[Dict]:
        """解析JSON响应"""
        try:
            cleaned = self.clean_json_response(response_text)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            json_str = self.extract_json_object(response_text)
            if json_str:
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"✗ JSON parse error: {e}")
            else:
                print(f"✗ No JSON found in response")
            return None
    
    def build_message_content(self, text: str, image_paths: List[str]) -> List[Dict]:
        """构建消息内容"""
        content = [{"type": "text", "text": text}]
        for img_path in image_paths:
            img_base64 = self.encode_image_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })
        return content
    
    def call_api(self, prompt: str, image_paths: List[str]) -> Optional[Dict]:
        """调用API"""
        try:
            payload = {
                "model": self.config.model,
                "messages": [{
                    "role": "user",
                    "content": self.build_message_content(prompt, image_paths)
                }],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=self.config.get_headers(),
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                print(f"✗ API error: {response.status_code}")
                print(f"✗ Response: {response.text[:500]}")
                return None
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            return self.parse_json_response(content)
            
        except requests.exceptions.Timeout:
            print(f"✗ API timeout after {self.config.timeout}s")
            return None
        except Exception as e:
            print(f"✗ API call failed: {e}")
            return None
    
    def validate_fields(self, response: Dict, required_fields: List[str]) -> bool:
        """验证响应字段"""
        missing = [f for f in required_fields if f not in response]
        if missing:
            print(f"✗ Response missing fields: {', '.join(missing)}")
            print(f"✗ Received fields: {list(response.keys())}")
            return False
        return True
    
    @abstractmethod
    def validate_response(self, response: Dict) -> bool:
        """验证响应（子类实现）"""
        pass
