import sys
import types
import unittest
from unittest.mock import patch

from tradingagents.deepseek import DeepSeekChatClient


class DeepSeekChatClientTests(unittest.TestCase):
    def test_adapter_uses_openai_compatible_messages_without_exposing_key(self):
        calls = {}

        class FakeCompletions:
            def create(self, **kwargs):
                calls.update(kwargs)
                message = types.SimpleNamespace(content="  测试返回  ")
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)]
                )

        class FakeOpenAI:
            def __init__(self, **kwargs):
                calls["client"] = kwargs
                self.chat = types.SimpleNamespace(completions=FakeCompletions())

        fake_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
        with patch.dict(sys.modules, {"openai": fake_module}):
            client = DeepSeekChatClient(api_key="unit-test-placeholder")
            response = client.invoke(
                [("system", "系统约束"), ("human", "证据 JSON")]
            )

        self.assertEqual(response.content, "测试返回")
        self.assertEqual(calls["messages"][0]["role"], "system")
        self.assertEqual(calls["messages"][1]["role"], "user")
        self.assertNotIn("unit-test-placeholder", repr(calls["messages"]))


if __name__ == "__main__":
    unittest.main()
