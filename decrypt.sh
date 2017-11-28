# Simple and bad way to handle passwords, but this is a temporary workaround until
# I figure out how to do attribute based authentication

openssl enc -aes-256-cbc -d -in passwords.txt -out passwords1.txt -pass pass:cpsc454

mv passwords1.txt passwords.txt
