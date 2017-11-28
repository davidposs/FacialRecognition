# Basic and awful way of storing passwords, but it is a workaround until 
# I figure out how to do attribute based authentication

openssl enc -aes-256-cbc -in passwords.txt -out passwords1.txt -pass pass:cpsc454

mv passwords1.txt passwords.txt
