# -*- coding: utf-8 -*-
""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
from unidecode import unidecode
from .numbers import normalize_numbers
from sys import version_info
import warnings
import hgtk

Cache = {}

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)

def _cosonant_and_vowel(string):
    retval = []

    for char in string:
      codepoint = ord(char)

      if codepoint < 0x80:  # Basic ASCII
        retval.append(str(char))
        continue

      if codepoint >= 0xac00 and codepoint <= 0xd7a3:
        kr_decomposed = hgtk.letter.decompose(char)
        retval += list(kr_decomposed)
        continue

      if codepoint > 0xeffff:
        continue  # Characters in Private Use Area and above are ignored

      if 0xd800 <= codepoint <= 0xdfff:
        warnings.warn("Surrogate character %r will be ignored. "
                      "You might be using a narrow Python build." % (char,),
                      RuntimeWarning, 2)

      section = codepoint >> 8  # Chop off the last two hex digits
      position = codepoint % 256  # Last two hex digits

      try:
        table = Cache[section]
      except KeyError:
        try:
          mod = __import__('unidecode.x%03x' % (section), globals(), locals(), ['data'])
        except ImportError:
          Cache[section] = None
          continue  # No match: ignore this character and carry on.

        Cache[section] = table = mod.data

      if table and len(table) > position:
        retval.append(table[position])

    return ''.join(retval)

def convert_to_cosonant_and_vowel(text):
  _warn_if_not_unicode(text)
  try:
    bytestring = text.encode('ASCII')
  except UnicodeEncodeError:
    return _cosonant_and_vowel(text)
  if version_info[0] >= 3:
    return text
  else:
    return bytestring


def _warn_if_not_unicode(string):
    if version_info[0] < 3 and not isinstance(string, unicode):
      warnings.warn("Argument %r is not an unicode object. "
                    "Passing an encoded string will likely have "
                    "unexpected results." % (type(string),),
                    RuntimeWarning, 2)

def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def english_cleaners(text):
  '''Pipeline for English text, including number and abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)
  return text

def korean_cleaners(text):
  '''Pipeline for English text, including number and abbreviation expansion.'''
  text = convert_to_cosonant_and_vowel(text)
  text = lowercase(text)
  text = expand_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)
  return text