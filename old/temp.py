import fitz
print(fitz.__doc__)  # 如果是 PyMuPDF，会显示文档说明
doc = fitz.open(r"C:\Users\70739\Desktop\保研\北京邮电大学-王小捷组\2463_Controlled_Low_Rank_Adapt.pdf")
  # 试试能不能打开一个本地PDF文件
print(doc.page_count)
