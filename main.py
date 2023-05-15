# -*- coding:utf-8 -*-
import os
import tornado.web
import tornado.ioloop
from tornado.options import define, options
from web_handler import ChatHandler, UpsertHandler

define("port", default=9000, help=" run on this port", type=int)


def main():
    tornado.options.parse_command_line()
    application = tornado.web.Application(
        [

            # chat
            (r"/chat", ChatHandler),

            # upsert
            (r"/upsert", UpsertHandler),

        ],
        template_path=os.path.join(os.path.dirname(__file__), "html_tpl"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        debug=False
    )
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(options.port)
    # tornado.httpclient.AsyncHTTPClient.configure("tornado.curl_httpclient.CurlAsyncHTTPClient")
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
