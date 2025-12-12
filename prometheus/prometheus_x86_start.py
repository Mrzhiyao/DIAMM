import subprocess
import time
import logging

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kubectl_monitor.log"),
        logging.StreamHandler()
    ]
)

def apply_yaml():
    """执行kubectl apply命令并记录结果"""
    command = "kubectl apply -f dcgm-exporter-servicemonitor.yaml"
    try:
        # 执行命令并捕获输出
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=30  # 设置30秒超时
        )
        logging.info(f"Command executed successfully:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with return code {e.returncode}:\n{e.stderr}")
    except subprocess.TimeoutExpired:
        logging.error("Command timed out after 30 seconds")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    
    return False

def main():
    """主循环，每分钟执行一次命令"""
    logging.info("Starting automated kubectl apply scheduler")
    success_count = 0
    failure_count = 0
    
    try:
        while True:
            success = apply_yaml()
            if success:
                success_count += 1
            else:
                failure_count += 1
                
            logging.info(f"Statistics: {success_count} successes, {failure_count} failures")
            logging.info("Sleeping for 60 seconds...")
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("\nProgram terminated by user. Final statistics:")
        logging.info(f"Total runs: {success_count + failure_count}")
        logging.info(f"Successes: {success_count} | Failures: {failure_count}")
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        logging.info("Program terminated unexpectedly")

if __name__ == "__main__":
    main()
