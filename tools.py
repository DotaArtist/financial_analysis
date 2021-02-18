#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""功能函数"""

__author__ = 'yp'


import pdfminer.high_level


def transform_pdf_str(pdf_file):
    """读取PDF"""
    return pdfminer.high_level.extract_text(pdf_file=pdf_file)
