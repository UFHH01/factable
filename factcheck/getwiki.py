"""
    getwiki.py
"""

import wikipedia

wikipedia.set_lang('en')

ny = wikipedia.page('New York')
print(ny.content)
