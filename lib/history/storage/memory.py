from datetime import datetime

from lib.history.storage.abstract import ChatHisoryAbstractStorage
from lib.utils.log import create_log_dir, save_log


class ChatHistoryMemoryStorage(ChatHisoryAbstractStorage):
    def __init__(self, uid: str, manifest_key: str):
        self.__messages = []
        self.uid = uid
        self.manifest_key = manifest_key

        self.time = datetime.now()
        # init log dir
        create_log_dir(manifest_key)

    def append(self, data):
        self.__messages.append(data)
        save_log(self.manifest_key, self.messages(), self.time)

    def get(self, index):
        return self.__messages[index]

    def get_data(self, index, name):
        m = self.__messages[index]
        if m:
            return m.get(name)

    def set(self, index, data):
        if self.__messages[index]:
            self.__messages[index] = data

    def len(self):
        return len(self.__messages)

    def last(self):
        if self.len() > 0:
            return self.__messages[self.len() - 1]

    def messages(self):
        return self.__messages

    def restore(self, data):
        self.__messages = data
