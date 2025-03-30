 """
测试夸克搜索工具
"""
import pytest
import os
from dotenv import load_dotenv
from src.tools import QuarkSearchTool

def load_config():
    """加载环境变量配置"""
    load_dotenv()
    return {
        "ak": os.getenv("QUARK_API_AK"),
        "sk": os.getenv("QUARK_API_SK")
    }

@pytest.fixture
def quark_search_tool():
    """创建夸克搜索工具实例"""
    config = load_config()
    if not config["ak"] or not config["sk"]:
        pytest.skip("未设置 QUARK_AK 或 QUARK_SK 环境变量")
    return QuarkSearchTool(
        ak=config["ak"],
        sk=config["sk"]
    )

def test_quark_search(quark_search_tool):
    """测试夸克搜索功能"""
    # 测试搜索
    query = "Python编程"
    result = quark_search_tool.run(query)

    # 验证结果不为空
    assert result is not None
    assert isinstance(result, str)

    # 验证结果格式
    import json
    try:
        results = json.loads(result)
        assert isinstance(results, list)
        if len(results) > 0:
            # 验证结果项的结构
            first_result = results[0]
            assert "title" in first_result
            assert "url" in first_result
            assert "snippet" in first_result
    except json.JSONDecodeError:
        pytest.fail("结果不是有效的JSON格式")

@pytest.mark.asyncio
async def test_quark_search_async(quark_search_tool):
    """测试异步夸克搜索功能"""
    query = "Python异步编程"
    result = await quark_search_tool.arun(query)

    # 验证结果不为空
    assert result is not None
    assert isinstance(result, str)

    # 验证结果格式
    import json
    try:
        results = json.loads(result)
        assert isinstance(results, list)
        if len(results) > 0:
            # 验证结果项的结构
            first_result = results[0]
            assert "title" in first_result
            assert "url" in first_result
            assert "snippet" in first_result
    except json.JSONDecodeError:
        pytest.fail("结果不是有效的JSON格式")

def test_quark_search_error_handling(quark_search_tool):
    """测试错误处理"""
    # 测试空查询
    result = quark_search_tool.run("")
    assert "搜索失败" in result or "搜索出错" in result

    # 测试特殊字符
    result = quark_search_tool.run("!@#$%^&*()")
    assert result is not None
    assert isinstance(result, str)