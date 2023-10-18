#!/usr/bin/env python3
# python -m samples.SimpleTNE
import os
import sys
import yaml
import platform

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
# TODO: status checks
# from status import Status

from db_connectors import PostgresConnector
from prompt_formatters import RajkumarFormatter
from slashgpt.chat_config import ChatConfig  # noqa: E402
from slashgpt.chat_session import ChatSession  # noqa: E402
from slashgpt.utils.print import print_bot, print_error, print_function, print_info  # noqa: E402

if platform.system() == "Darwin":
    # So that input can handle Kanji & delete
    import readline  # noqa: F401

DB_PORT = 5432
DB_NAME = "ebp"
DB_USER = "postgres"
DB_PASS = "i6XFDgR6KGkTd"
DB_HOST = "postgresql-ebp.cfwmuvh4blso.us-west-2.rds.amazonaws.com"

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

class SimpleSlashGPTServer:

    def __init__(self, manifest_path: str, config_path: Optional[str]):

        # Load the chat session config
        # TODO: status
        # self.status = status.INITIALIZING
        if not config_path:
            self.path = os.path.join(os.path.dirname(__file__), "../..")
        self.config = ChatConfig(self.path)

        # Read the manifest
        with open(sys.argv[1], 'r') as fp:
            self.manifest = yaml.safe_load(fp)

        # Pull the database metadata into the prompt
        schema = self.get_schema()
        # TODO: this shouldn't always append to the front, it should overwrite if already exists
        self.manifest['prompt'].insert(0, schema)

        # TODO: status
        # self.status = status.HEALTHY

    def get_schema(self):
        db_name = self.manifest.get('database')
        if not db_name:
            raise ValueError("Manifest must specify a database")

        # Pull schema
        db_connector = self.connect_db(db_name)
        prompt_engineer = self.get_formatter(db_connector)
        schema = prompt_engineer.pull_schema()

        return schema

    def connect_db(self, db_name: str):
        # TODO: load all these from environment variables specified in manifest
        db_connector = PostgresConnector(
            user=DB_USER, password=DB_PASS, dbname=db_name, host=DB_HOST, port=DB_PORT
        )

        db_connector.connect()
        return db_connector

    # TODO: db_connector typing
    def get_formatter(self, db_connector):
        db_schema = [
            db_connector.get_schema(table) for table in db_connector.get_tables()
        ]
        db_data = [
            db_connector.select_three(table) for table in db_connector.get_tables()
        ]
        db_unique_vals = {
            table: db_connector.get_distinct_values(table)
            for table in db_connector.get_tables()
        }

        formatter = RajkumarFormatter(db_schema, db_data, db_unique_vals)

        return formatter

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "../..")
    config = ChatConfig(path)

    # Read the manifest
    with open(sys.argv[1], 'r') as fp:
        manifest = yaml.safe_load(fp)

    print(manifest)
    breakpoint()

    main = SimpleTNE(config, manifest, "TNE")
    main.start()
