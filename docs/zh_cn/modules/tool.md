# 自定义工具

AgentLego 是可扩展的，您可以轻松地添加自定义工具并将其应用于各种智能体系统。

## 简单示例

首先，所有工具都应该继承 `BaseTool` 类。以一个简单的时钟工具为例。

```python
from agentlego.tools import BaseTool
from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta

class Clock(BaseTool):
    def __init__(self):
        toolmeta = ToolMeta(
            name='Clock',
            description='返回当前日期和时间的时钟。',
            inputs=[],
            outputs=['text'],
        )
        super().__init__(toolmeta=toolmeta, parser=DefaultParser)
```

在初始化方法中，您需要构建一个 `ToolMeta` 来指定名称、描述、输入参数类别和输出类别。目前可用的类别有 `text`、`image` 和 `audio`。

然后，还需要指定一个默认解析器 (parser)，它用于处理输入和输出类型。通常情况下，您可以直接使用 `DefaultParser` 作为默认解析器。

之后，可以重载 `BaseTool` 的 `setup` 和 `apply` 方法。`setup` 方法将在第一次调用工具时运行，通常用于延迟加载一些重型模块。`apply` 方法是在调用工具时执行的核心方法。在这个示例中，我们只需重载 `apply` 方法。

```python
class Clock(BaseTool):
    ...

    def apply(self):
        from datetime import datetime
        return datetime.now().strftime('%Y/%m/%d %H:%M')
```

我们已经完成了这个工具，现在您可以实例化它并在智能体系统中使用。

```python
# 创建一个实例
tool = Clock()

# 在 langchain 中使用
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

agent = initialize_agent(
    agent='structured-chat-zero-shot-react-description',
    llm=ChatOpenAI(temperature=0.),
    tools=[tool.to_langchain()],
    verbose=True)
agent.invoke("现在几点了？")

# 在 lagent 中使用
from lagent import ReAct, GPTAPI, ActionExecutor
agent = ReAct(
    llm=GPTAPI(temperature=0.),
    action_executor=ActionExecutor([tool.to_lagent()])
)
ret = agent.chat("现在几点了？")
print(ret.response)
```

## 多模态输入和输出

AgentLego 的一个核心特性是支持多模态工具，同时我们也需要处理多模态的输入和输出。不同的智能体系统接受不同的输入输出格式，
通常基于语言的智能体系统无法直接接受图像和音频，我们需要将输入和输出数据转换为字符串，比如文件路径。
但是一些智能体系统可以直接处理原始数据，比如Transformers Agent，还有一些智能体系统则需要原始数据用以在前端展示。

因此，我们使用代理类型作为工具的输入和输出类型，并使用一个`parser`自动将其转换为目标格式。

假设我们要实现一个工具，它可以使用输入图像，用指定语言生成一个音频概述。

```python
from agentlego.tools import BaseTool
from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import ImageIO, AudioIO

class AudioCaption(BaseTool):
    def __init__(self):
        toolmeta = ToolMeta(
            name='AudioCaption',
            description='一个可以根据输入图像和指定语言，生成概要音频的工具。',
            inputs=['image', 'text'],
            outputs=['audio'],
        )
        super().__init__(toolmeta=toolmeta, parser=DefaultParser)

    def apply(self, image: ImageIO, language: str):
        # 将代理类型转换为工具中所需的格式。
        image = image.to_pil()

        # 处理流程伪代码
        caption = image_caption(image)
        caption = translate(caption, language)
        audio_tensor = text_reader(caption, language)

        # 返回封装后的代理类型
        return AudioIO(audio_tensor)
```

代理类型可以直接转换为支持的格式，也可以从支持的格式直接构造代理类型。
默认的 `DefaultParser` 会将输入转换为代理类型，并将所有输出代理类型转换为字符串格式输出。
因此，您可以直接调用工具并使用文件路径作为参数。

```python
>>> tool = AudioCaption()
>>> audio_path = tool('examples/demo.png', 'en-US')
>>> print(audio_path)
generated/audio/20231011-1730.wav
```
