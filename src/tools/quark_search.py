"""
Tool for searching using Alibaba Quark Browser API.
"""

from typing import Optional, Type
from langchain.tools import BaseTool
import requests
import json
import os
from datetime import datetime
import hashlib
import base64

class QuarkSearchTool(BaseTool):
    name = "quark_search"
    description = "使用阿里夸克浏览器API进行搜索。输入应该是一个搜索查询。"
    
    def __init__(self, ak: str, sk: str):
        super().__init__()
        self.ak = ak
        self.sk = sk
        self.base_url = "https://api.quark.cn/search"
    
    def _get_signature(self, params: dict) -> str:
        """生成签名"""
        # 按字母顺序排序参数
        sorted_params = sorted(params.items(), key=lambda x: x[0])
        # 构建签名字符串
        sign_str = "&".join([f"{k}={v}" for k, v in sorted_params])
        # 添加SK
        sign_str = f"{sign_str}&{self.sk}"
        # 计算MD5
        md5 = hashlib.md5(sign_str.encode()).hexdigest()
        return md5
    
    def _run(self, query: str) -> str:
        """运行搜索"""
        try:
            # 准备请求参数
            params = {
                "ak": self.ak,
                "timestamp": int(datetime.now().timestamp()),
                "query": query,
                "page": 1,
                "size": 10
            }
            
            # 添加签名
            params["sign"] = self._get_signature(params)
            
            # 发送请求
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            # 格式化结果
            if result.get("code") == 0:
                items = result.get("data", {}).get("items", [])
                formatted_results = []
                for item in items:
                    formatted_results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", "")
                    })
                return json.dumps(formatted_results, ensure_ascii=False, indent=2)
            else:
                return f"搜索失败: {result.get('message', '未知错误')}"
                
        except Exception as e:
            return f"搜索出错: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """异步运行搜索"""
        # 由于requests库不支持异步，这里直接调用同步方法
        return self._run(query) 