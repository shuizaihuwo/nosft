# convert_vulnerabilities.py

import csv

def convert_csv_to_txt(csv_file_path, txt_file_path):
    """
    将漏洞信息的CSV文件转换为TXT格式，并添加描述信息。

    参数:
    - csv_file_path: 输入的CSV文件路径。
    - txt_file_path: 输出的TXT文件路径。
    """
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile, \
             open(txt_file_path, 'w', encoding='utf-8') as txtfile:
            
            reader = csv.DictReader(csvfile)
            
            # 写入文件的描述或介绍
            txtfile.write("漏洞信息知识图谱数据集\n")
            txtfile.write("=====================\n\n")
            txtfile.write("以下内容包含了多个漏洞的信息，每个漏洞的信息以详细描述的形式呈现，便于构建知识图谱。\n\n")
            
            for row in reader:
                txtfile.write("-----\n")
                txtfile.write(f"CVE ID: {row['cve_id']}\n")
                txtfile.write(f"厂商/项目: {row['vendor_project']}\n")
                txtfile.write(f"产品: {row['product']}\n")
                txtfile.write(f"漏洞名称: {row['vulnerability_name']}\n")
                txtfile.write(f"添加日期: {row['date_added']}\n")
                txtfile.write(f"简要描述: {row['short_description']}\n")
                txtfile.write(f"所需措施: {row['required_action']}\n")
                txtfile.write(f"截止日期: {row['due_date']}\n")
                txtfile.write(f"备注: {row['notes']}\n")
                txtfile.write(f"组别: {row['grp']}\n")
                txtfile.write(f"公开日期: {row['pub_date']}\n")
                txtfile.write(f"CVSS评分: {row['cvss']}\n")
                txtfile.write(f"CWE编号: {row['cwe']}\n")
                txtfile.write(f"向量: {row['vector']}\n")
                txtfile.write(f"复杂度: {row['complexity']}\n")
                txtfile.write(f"严重性: {row['severity']}\n")
                txtfile.write("-----\n\n")
        
        print(f"成功将CSV文件转换为TXT格式并保存到 {txt_file_path}")
    
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file_path}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 输入的CSV文件路径
    csv_file = "/home/qy/keyan/nosft/data/raw_data/2022-06-08-enriched.csv"
    
    # 输出的TXT文件路径
    txt_file = "/home/qy/keyan/nosft/data/raw_data/2022-06-08-enriched.txt"
    
    convert_csv_to_txt(csv_file, txt_file)
