import http.server
import socketserver
import json
import urllib.request
import urllib.parse
from pathlib import Path
import os
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).parent.parent
LEVELS_DIR = PROJECT_ROOT / "levels"
PORT = 8080

class GDLevelManagerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
        
    def do_POST(self):
        if self.path == '/api/save':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            level_name = data.get('level_name', 'Unknown')
            full_id = data.get('full_id')
            layout_id = data.get('layout_id')
            theme = data.get('theme', 'Hell')
            
            try:
                folder_name = self.save_level(level_name, full_id, layout_id, theme)
                
                # trigger build datasets optionally
                import subprocess
                subprocess.run(["python", str(PROJECT_ROOT / "build_datasets.py"), folder_name], cwd=str(PROJECT_ROOT))
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode())
        else:
            self.send_error(404)

    def download_level_data(self, level_id):
        url = "http://www.boomlings.com/database/downloadGJLevel22.php"
        post_data = {"gameVersion": "22", "binaryVersion": "38", "gdw": "0", "secret": "Wmfd2893gb7", "levelID": str(level_id)}
        encoded_data = urllib.parse.urlencode(post_data).encode("utf-8")
        req = urllib.request.Request(url, data=encoded_data, headers={'User-Agent': ''})
        resp = urllib.request.urlopen(req)
        body = resp.read().decode('utf-8')
        if body.startswith('-1'):
            raise Exception(f"Level {level_id} download failed (maybe unlisted or invalid)")
        
        parts = body.split(':')
        for i in range(len(parts)):
            if parts[i] == '4':
                return parts[i+1] # The level data base64
        raise Exception("Level data not found in response")

    def create_gmd_xml(self, name, level_string):
        root = ET.Element("plist", version="1.0", gjver="2.0")
        dict_node = ET.SubElement(root, "dict")
        
        k2 = ET.SubElement(dict_node, "k")
        k2.text = "k2"
        s2 = ET.SubElement(dict_node, "s")
        s2.text = name
        
        k4 = ET.SubElement(dict_node, "k")
        k4.text = "k4"
        s4 = ET.SubElement(dict_node, "s")
        s4.text = level_string
        
        # Just create simple xml string
        return ET.tostring(root, encoding="utf-8", method="xml")

    def save_level(self, name, full_id, layout_id, theme):
        LEVELS_DIR.mkdir(exist_ok=True)
        # safe folder name
        safe_name = "".join([c for c in name if c.isalnum() or c in (" ", "_")]).replace(" ", "")
        if not safe_name: safe_name = f"Level_{full_id}"
        level_dir = LEVELS_DIR / safe_name
        level_dir.mkdir(exist_ok=True)
        
        full_data = self.download_level_data(full_id)
        layout_data = self.download_level_data(layout_id)
        
        with open(level_dir / "full.gmd", "wb") as f:
            f.write(self.create_gmd_xml(name, full_data))
            
        with open(level_dir / "layout.gmd", "wb") as f:
            f.write(self.create_gmd_xml(f"{name} Layout", layout_data))
            
        with open(level_dir / "theme.txt", "w", encoding="utf-8") as f:
            f.write(theme)
            
        return safe_name

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), GDLevelManagerHandler) as httpd:
        print(f"Serving UI at http://localhost:{PORT}")
        httpd.serve_forever()
