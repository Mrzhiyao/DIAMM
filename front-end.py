import http.server
import socketserver

PORT = 8989

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':  # 根路径
            self.path = 'ui_mmtest_enn.html'  # 重定向到 abc.html
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
    print(f"Serving at http://0.0.0.0:{PORT}")
    httpd.serve_forever()
