main_backup_5_1_2.py
# ====================================================
# inspect_columns.py
# 功能：检查整篇IEEE双栏PDF的文字提取效果
# 不调用Claude API，不产生模型费用
# ====================================================

# 导入Python路径处理工具
from pathlib import Path

# 导入PyMuPDF PDF解析库
import pymupdf

# 导入文件选择窗口
from tkinter import Tk, filedialog


# ================== 1. 选择PDF文件 ==================

# 创建文件选择窗口
root = Tk()

# 隐藏主窗口
root.withdraw()

# 弹出文件选择对话框
pdf_path = filedialog.askopenfilename(
    title="请选择需要检查的IEEE论文PDF",
    filetypes=[("PDF文件", "*.pdf")]
)

# 关闭窗口资源
root.destroy()

# 没有选择文件则结束
if not pdf_path:
    raise SystemExit("未选择PDF，程序已结束。")


# ================== 2. 按左右栏提取文字 ==================

def extract_columns(page):

    # 获取当前页面的矩形范围
    rect = page.rect

    # 获取页面中间位置
    middle = rect.x0 + rect.width / 2

    # 左栏矩形
    left_rect = pymupdf.Rect(
        rect.x0,
        rect.y0,
        middle,
        rect.y1
    )

    # 右栏矩形
    right_rect = pymupdf.Rect(
        middle,
        rect.y0,
        rect.x1,
        rect.y1
    )

    # 提取左栏文字
    left_text = page.get_text(
        "text",
        clip=left_rect,
        sort=True
    )

    # 提取右栏文字
    right_text = page.get_text(
        "text",
        clip=right_rect,
        sort=True
    )

    # 返回左右栏文字
    return left_text.strip(), right_text.strip()


# ================== 3. 读取整篇PDF ==================

# 打开PDF
doc = pymupdf.open(pdf_path)

# 获取PDF总页数
total_pages = len(doc)

# 在项目目录下创建报告文件夹
output_dir = Path("column_reports")

# 若目录不存在则自动创建
output_dir.mkdir(exist_ok=True)

# 保存详细检查内容
all_reports = []

# 保存页面统计信息
summary_lines = [
    "页码\t原始字符数\t左栏字符数\t右栏字符数\t差值"
]


# ================== 4. 遍历所有页面 ==================

for page_number in range(1, total_pages + 1):

    # 获取当前页，PDF索引从0开始
    page = doc[page_number - 1]

    # 普通文字提取
    original_text = page.get_text(
        "text",
        sort=True
    ).strip()

    # 分别提取左右栏
    left_text, right_text = extract_columns(page)

    # 合并左右栏文本
    column_text = left_text + "\n\n" + right_text

    # 计算字符数
    original_count = len(original_text)
    left_count = len(left_text)
    right_count = len(right_text)

    # 字符数量差值
    difference = (
        original_count - left_count - right_count
    )

    # 保存当前页统计数据
    summary_lines.append(
        f"{page_number}\t"
        f"{original_count}\t"
        f"{left_count}\t"
        f"{right_count}\t"
        f"{difference}"
    )

    # 整理当前页详细报告
    report = (
        f"\n{'=' * 65}\n"
        f"PDF第{page_number}页\n"
        f"{'=' * 65}\n\n"
        f"【原始文本】\n{original_text}\n\n"
        f"【左栏文本】\n{left_text}\n\n"
        f"【右栏文本】\n{right_text}\n"
    )

    # 保存到总报告
    all_reports.append(report)

    # 每页单独保存一份
    page_file = output_dir / f"page_{page_number:02d}.txt"

    page_file.write_text(
        report,
        encoding="utf-8"
    )

    # 终端显示检查进度
    print(
        f"第{page_number}/{total_pages}页完成，"
        f"左栏{left_count}字符，"
        f"右栏{right_count}字符"
    )


# ================== 5. 保存报告 ==================

# 保存完整报告
(output_dir / "full_report.txt").write_text(
    "\n".join(all_reports),
    encoding="utf-8"
)

# 保存统计数据
(output_dir / "summary.tsv").write_text(
    "\n".join(summary_lines),
    encoding="utf-8-sig"
)

# 关闭PDF
doc.close()

# 输出结果
print("\n全部页面检查完成！")
print("PDF页数：", total_pages)
print("报告目录：", output_dir.resolve())
print("完整报告：column_reports/full_report.txt")
print("统计文件：column_reports/summary.tsv")