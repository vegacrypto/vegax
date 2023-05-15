"""
功能：langchain主工程
版本：1.0
日期：2023/05/15
开发者：Peter
"""
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
import tornado.ioloop
import json
from logger.mylogger import Logger
from define import ResCode
from chat_manager import ChatManager


rescode = ResCode()
log_main = Logger.get_logger(__file__)


class ChatHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(4)

    # @tornado.web.asynchronous
    @tornado.gen.coroutine
    def post(self):
        try:
            json_data = json.loads(self.request.body)
            # chat_model = str(json_data["model"])
            chat_scene = str(json_data["scene"])
            chat_message = str(json_data["chat"])
            log_main.info('Chat In Sence Work! Chat :{},Scene:{}'.format(chat_scene, chat_message))
        except Exception as e:
            result = dict()
            result["code"] = rescode.ILLEGAL_INPUT
            result["message"] = str(e)
            log_main.error('Illegal_input....')
            self.write(result)
            return
        try:
            cm = ChatManager()
            chat_response = cm.chat_with_template(chat_scene, chat_message)
            # tornado.ioloop.IOLoop.instance().add_callback(self.my_gen_chord, chat_scene, chat_message)
            result = dict()
            result["code"] = rescode.OK
            result["message"] = "OK"
            result["AI_response"] = chat_response
            log_main.info('code:{}_AI_response:{}'.format(rescode.OK, chat_response))
            self.write(result)
            return
        except Exception as e:
            result = dict()
            result["code"] = rescode.UNKNOW_ERROR
            result["message"] = str(e)
            log_main.error('code:{}_error:{}'.format(rescode.UNKNOW_ERROR, str(e)))
            self.write(result)
            return

    @run_on_executor
    def my_gen_chord(self, chat_scene, chat_message):
        try:
            cm = ChatManager()
            chat_response = cm.chat_with_template(chat_scene, chat_message)
            result = dict()
            result["code"] = rescode.OK
            result["message"] = "OK"
            result["AI_response"] = chat_response
            log_main.info('code:{}_AI_response:{}'.format(rescode.OK, chat_response))
            self.write(result)
            return
        except Exception as e:
            result = dict()
            result["code"] = rescode.UNKNOW_ERROR
            result["message"] = str(e)
            log_main.error('code:{}_error:{}'.format(rescode.UNKNOW_ERROR, str(e)))
            self.write(result)
            return


class UpsertHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(4)

    # @tornado.web.asynchronous
    @tornado.gen.coroutine
    def post(self):
        try:
            json_data = json.loads(self.request.body)
            log_main.info('Hahaha'.format(str(json_data)))
        except Exception as e:
            result = dict()
            result["code"] = rescode.ILLEGAL_INPUT
            result["message"] = str(e)
            log_main.error('Illegal_input....')
            self.write(result)
            return
