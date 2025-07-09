import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

WATCH_PATH = "./"  # 监听当前目录

class RestartHandler(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.restart()

    def restart(self):
        if self.process:
            self.process.kill()
            time.sleep(1)
        print("[INFO] 🔄 正在重启 Gradio 应用...")
        self.process = subprocess.Popen(["python", "gradio_app.py"])

    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            print(f"[INFO] 检测到修改：{event.src_path}")
            self.restart()

if __name__ == "__main__":
    event_handler = RestartHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_PATH, recursive=True)
    observer.start()
    print("[INFO] ✅ 正在监听代码改动...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[INFO] ❌ 停止监听")
    observer.join()
