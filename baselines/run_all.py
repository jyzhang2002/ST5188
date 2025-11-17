import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"start: {script_name}")
    print(f"{'='*50}")
    
    try:
        # check if script exists
        if not os.path.exists(script_name):
            print(f"Error: file {script_name} does not exist.")
            return False
        
        # run script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True)
        
        # print output
        if result.stdout:
            print("output:")
            print(result.stdout)
        
        if result.returncode != 0:
            print(f"Error: {script_name} fail")
            if result.stderr:
                print("info:")
                print(result.stderr)
            return False
        else:
            print(f"✓ {script_name} success")
            return True
            
    except Exception as e:
        print(f"run {script_name} error: {e}")
        return False

def main():
    # scripts = ["dlinear.py", "gru.py", "lstm.py"]
    # scripts = ["lstm.py"]
    scripts = ["deepar.py", "ConvTrans.py"]
    
    print("running...")
    
    success_count = 0
    for script in scripts:
        if run_script(script):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"finish: success {success_count}/{len(scripts)} scripts")
    print(f"{'='*50}")
    
    if success_count == len(scripts):
        print("🎉 all scripts success！")
    else:
        print(f"⚠️  {len(scripts) - success_count} script(s) failed.")

if __name__ == "__main__":
    main()