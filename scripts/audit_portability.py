import os
import re

# Emojis that might cause CP949 errors
target_emojis = ["🔧", "📏", "📐", "🎯", "🌍", "▶️", "🥇", "🔍", "🚀", "📷", "✅"]
# Absolute paths regex (e.g., E:\ or C:\ or /home/)
abs_path_re = re.compile(r'[A-Za-z]:\\[^ \n]+|/[^ \n]+')

def audit_file(filepath):
    results = {"emojis": [], "abs_paths": []}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # Check for emojis
                for em in target_emojis:
                    if em in line:
                        results["emojis"].append((i+1, line.strip()))
                
                # Check for absolute paths
                matches = abs_path_re.findall(line)
                for m in matches:
                    # Ignore some common false positives like metadata artifacts or file URLs
                    if "file:///" in m or ".gemini" in m or ".git" in m:
                        continue
                    if m.startswith("C:\\Users") or m.startswith("e:\\"):
                        results["abs_paths"].append((i+1, m))
    except Exception as e:
        pass
    return results

root = "."
for dirpath, dirnames, filenames in os.walk(root):
    if ".git" in dirpath or "__pycache__" in dirpath:
        continue
    for f in filenames:
        if f.endswith(".py") or f.endswith(".md"):
            path = os.path.join(dirpath, f)
            res = audit_file(path)
            if res["emojis"] or res["abs_paths"]:
                print(f"--- {path} ---")
                if res["emojis"]:
                    print("  Emojis found:")
                    for ln, text in res["emojis"]:
                        print(f"    L{ln}: {text}")
                if res["abs_paths"]:
                    print("  Absolute paths found:")
                    for ln, text in res["abs_paths"]:
                        print(f"    L{ln}: {text}")
