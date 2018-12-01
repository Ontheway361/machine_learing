#!/usr/bin/python
#-*- coding: utf-8 -*-

import re
import bayes

emailText = open('email/ham/6.txt', 'rb').read()

regEx = re.compile('\\W')

listOfTokens = re.split(emailText)
