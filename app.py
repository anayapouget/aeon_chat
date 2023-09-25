# Modified from https://github.com/hwchase17/chat-your-data

import os
import constants
from typing import Optional, Tuple
from threading import Lock

import gradio as gr

from query_data import get_condense_prompt_qa_chain

def parse_codeblock(text):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "```" in line:
            if line != "```":
                lines[i] = f'<pre><code class="{lines[i][3:]}">'
            else:
                lines[i] = '</code></pre>'
        else:
            if i > 0:
                lines[i] = "<br/>" + line
    return "".join(lines)

class ChatWrapper:

    def __init__(self):
        self.lock = Lock()

    def __call__(
        self, inp: str, history: Optional[Tuple[str, str]], chain
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            if chain is None:
                os.environ["OPENAI_API_KEY"] = constants.APIKEY
                chain = get_condense_prompt_qa_chain()
            # Run chain and append input.
            output = chain({"question": inp})["answer"]
            output = parse_codeblock(output)
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history, chain


chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: white}")

with block:
    with gr.Row():
        gr.Markdown(
            "<h1><center>Aeon Chat</center></h1>")

    chatbot = gr.Chatbot().style(height=870)

    with gr.Row():
        message = gr.Textbox(
            show_label=False,
            placeholder="Ask questions about Project Aeon",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(
            full_width=False)

    gr.Examples(
        examples=[
            "What is the aeon_mecha github repo for?",
            "How could I load raw data?",
            "What do the Experiment tables in the database contain?",
            "How could I create a new Device made out of streams from patches and cameras?"
        ],
        inputs=message,
    )

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[message, state, agent_state], outputs=[chatbot, state, agent_state])
    message.submit(chat, inputs=[message, state, agent_state], outputs=[chatbot, state, agent_state])

block.launch(debug=True,show_api=False)