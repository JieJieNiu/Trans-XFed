#!/usr/bi/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 14:09:33 2023

@author: jolie
"""

from args import args_parser
from server_he import FedProx


def main():
    args = args_parser()
    fedProx = FedProx(args)
    fedProx.server()
    fedProx.global_test('test')


if __name__ == '__main__':
    main()