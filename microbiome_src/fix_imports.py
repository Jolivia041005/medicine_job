"""将所有 `from microbiome_src.` 替换为 `from microbiome_src.`"""
import pathlib

root = pathlib.Path("microbiome_src")
for py_file in root.rglob("*.py"):
    content = py_file.read_text(encoding="utf-8")
    if "from microbiome_src." in content:
        new_content = content.replace("from microbiome_src.", "from microbiome_src.")
        py_file.write_text(new_content, encoding="utf-8")
        print(f"✓ {py_file}")
print("替换完成。")