"""
PDF生成器 - 将HTML报告转换为PDF
"""

import subprocess
from pathlib import Path


def generate_pdf_report(html_path: str, pdf_path: str) -> None:
    """
    将HTML报告转换为PDF格式
    
    使用wkhtmltopdf工具进行转换
    
    Args:
        html_path: HTML文件路径
        pdf_path: 输出PDF文件路径
    """
    try:
        # 检查wkhtmltopdf是否安装
        try:
            subprocess.run(
                ["wkhtmltopdf", "--version"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # 如果wkhtmltopdf未安装，尝试使用weasyprint
            try:
                from weasyprint import HTML
                
                HTML(html_path).write_pdf(pdf_path)
                return
            except ImportError:
                raise RuntimeError(
                    "PDF生成需要安装 wkhtmltopdf 或 weasyprint。\n"
                    "安装方法:\n"
                    "  - wkhtmltopdf: brew install wkhtmltopdf (macOS) 或 apt-get install wkhtmltopdf (Linux)\n"
                    "  - weasyprint: pip install weasyprint"
                )

        # 使用wkhtmltopdf转换
        cmd = [
            "wkhtmltopdf",
            "--enable-local-file-access",
            "--page-size", "A4",
            "--margin-top", "10mm",
            "--margin-bottom", "10mm",
            "--margin-left", "10mm",
            "--margin-right", "10mm",
            "--encoding", "UTF-8",
            html_path,
            pdf_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"PDF生成失败: {result.stderr}")

    except Exception as e:
        raise RuntimeError(f"PDF生成错误: {str(e)}")


def generate_pdf_report_alternative(html_path: str, pdf_path: str) -> None:
    """
    备用PDF生成方法 - 使用pdfkit
    
    Args:
        html_path: HTML文件路径
        pdf_path: 输出PDF文件路径
    """
    try:
        import pdfkit
        
        options = {
            'page-size': 'A4',
            'margin-top': '10mm',
            'margin-right': '10mm',
            'margin-bottom': '10mm',
            'margin-left': '10mm',
            'encoding': "UTF-8",
            'enable-local-file-access': None
        }
        
        pdfkit.from_file(html_path, pdf_path, options=options)
        
    except ImportError:
        raise RuntimeError(
            "pdfkit未安装。请运行: pip install pdfkit\n"
            "同时需要安装wkhtmltopdf: brew install wkhtmltopdf (macOS)"
        )
    except Exception as e:
        raise RuntimeError(f"PDF生成错误: {str(e)}")

