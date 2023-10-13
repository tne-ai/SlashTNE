#!/usr/bin/env python3
# python -m samples.SimpleTNE
import os
import sys
import yaml
import platform

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from slashgpt.chat_config import ChatConfig  # noqa: E402
from slashgpt.chat_session import ChatSession  # noqa: E402
from slashgpt.utils.print import print_bot, print_error, print_function, print_info  # noqa: E402

if platform.system() == "Darwin":
    # So that input can handle Kanji & delete
    import readline  # noqa: F401

class SimpleTNE:
    def __init__(self, config: ChatConfig, manifest: dict, agent_name: str):
        self.session = ChatSession(config, manifest=manifest, agent_name=agent_name)
        print_info(f"Activating: {self.session.title()}")

        if self.session.intro_message:
            print_bot(self.session.botname(), self.session.intro_message)

    def process_llm(self, session):
        try:
            (res, function_call) = session.call_llm()

            if res:
                print_bot(self.session.botname(), res)

            if function_call:
                (
                    function_message,
                    function_name,
                    should_call_llm,
                ) = function_call.process_function_call(
                    session.manifest,
                    session.history,
                    None,
                )
                if function_message:
                    print_function(function_name, function_message)

                if should_call_llm:
                    self.process_llm()

        except Exception as e:
            print_error(f"Exception: Restarting the chat :{e}")

    def start(self):
        while True:
            question = input(f"\033[95m\033[1m{self.session.username()}: \033[95m\033[0m").strip()
            if question:
                self.session.append_user_question(self.session.manifest.format_question(question))
                self.process_llm(self.session)


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "../..")
    config = ChatConfig(path)
    with open(sys.argv[1], 'r') as fp:
        manifest = yaml.safe_load(fp)
    main = SimpleTNE(config, manifest, "TNE")
    main.start()
