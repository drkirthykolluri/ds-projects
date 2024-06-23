from sys import argv

script, user_name = argv
prompt = '*'

print(f"Hi, {user_name}, I am the {script} script.")
print("I would like to ask you a few questions:")
print(f"Do you like me?")
likes=input(prompt)

print(f"Where do you live?")
lives=input(prompt)

print(f"""
      Alright, So your name is {user_name}
      and you live in {lives}
       """)