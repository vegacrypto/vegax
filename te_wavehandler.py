import requests
import json


class ChatTester(object):

    def __init__(self, server_url):
        self.server_url = server_url + '/chat'
        # self.header = {"Content-Type": "application/x-www-form-urlencoded"}
        self.header = {"Content-Type": "application/octet-stream"}

    def llm_post(self, scene, chat):
        payload = {'scene': scene, "chat": chat}
        print(payload)
        response = requests.post(self.server_url, data=json.dumps(payload), headers=self.header, verify=False)
        return response.text if response.status_code == 200 else None


if __name__ == '__main__':
    chat_tester = ChatTester("http://127.0.0.1:9000")
    scene = "In this conversation, you're a rock star's assistant, and all the questions and technical vocabulary revolve around music"
    chat = "Explain the classification of rock"
    res = chat_tester.llm_post(scene, chat)
    print(res)
