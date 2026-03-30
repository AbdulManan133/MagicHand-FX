import re

with open('main_back.py', 'r') as f:
    content = f.read()

# Replace literal \n with actual newlines and \" with "
content = content.replace(r'\n', '\n').replace(r'\"', '"')

with open('main_back.py', 'w') as f:
    f.write(content)

print('Fixed!')
