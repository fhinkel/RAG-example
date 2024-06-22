# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureAISearchDataSource,
    AzureChatCompletion,
    AzureChatPromptExecutionSettings,
    ExtraBody,
)
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.functions import KernelFunction
from semantic_kernel.connectors.memory.azure_cognitive_search.azure_ai_search_settings import AzureAISearchSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig

kernel = Kernel()
# Depending on the index that you use, you might need to enable the below
# and adapt it so that it accurately reflects your index.

# azure_ai_search_settings["fieldsMapping"] = {
#     "titleField": "source_title",
#     "urlField": "source_url",
#     "contentFields": ["source_text"],
#     "filepathField": "source_file",
# }

import sys
import os

current_dir = os.path.abspath('')
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
api_key = os.getenv("AZURE_AISEARCH_API_KEY")

# Create the data source settings
azure_ai_search_settings = AzureAISearchSettings(
    endpoint="https://sckw.search.windows.net", 
    index_name="ewog-index", 
    api_key=api_key)

az_source = AzureAISearchDataSource.from_azure_ai_search_settings(azure_ai_search_settings=azure_ai_search_settings)
extra = ExtraBody(data_sources=[az_source])

kernel = Kernel()

kernel.add_service(AzureChatCompletion(service_id="chat", env_file_path="../.env"))

chat_function = kernel.add_function(
    prompt="{{$chat_history}}{{$user_input}}",
    plugin_name="ChatBot",
    function_name="Chat",
)

execution_settings_extra = AzureChatPromptExecutionSettings(
    service_id="chat",
    max_tokens=2000,
    temperature=0.7,
    top_p=0.8,
    extra_body=extra,
)

history = ChatHistory()

history.add_assistant_message("Hi there, I'm the Fuzzy Friends of Endor customer service assistant. We love and sell live Ewogs. I'm curteous and helpful.")

arguments = KernelArguments(settings=execution_settings_extra)


async def handle_streaming(
    kernel: Kernel,
    chat_function: "KernelFunction",
    arguments: KernelArguments,
) -> None:
    response = kernel.invoke_stream(
        chat_function,
        return_function_results=False,
        arguments=arguments,
    )

    print("Fuzzy Friends customer service:> ", end="")
    result = ""
    async for message in response:
        print(str(message[0]), end="")
        result += str(message[0])
    print("\n")
    history.add_assistant_message(result)


async def chat() -> bool:
    try:
        user_input = input("User:> ")
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False
    arguments["user_input"] = user_input
    arguments["chat_history"] = history
    history.add_user_message(user_input)

    stream = True
    if stream:
        await handle_streaming(kernel, chat_function, arguments=arguments)
    else:
        result = await kernel.invoke(chat_function, arguments=arguments)
        print(f"Fuzzy Friends customer service:> {result}")
    return True


async def main() -> None:
    chatting = True
    print(
        "Welcome to Fuzzy Friends customer support!\
        \n  How may I help you?"
    )
    while chatting:
        chatting = await chat()


if __name__ == "__main__":
    asyncio.run(main())

