toxic_words=["stupid","boring","lame"]
comment=input("Enter a String:")
is_toxic=lambda comment:any(word in comment.lower() for word in toxic_words)
print(is_toxic(comment))
