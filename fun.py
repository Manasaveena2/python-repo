def perform_all_op(w):
    a=w.split()
    print(f"Total word count: {len(a)}")
    cnt=0
    s=0
    for i in a:
        if i.startswith('#'):
            cnt=cnt+1 
        s=s+len(i)
    print(f"Avg of word count: {s/len(a)}")
    print(f"Percentage of #: {(cnt/len(a))*100}")
   

w=input()
if w=='':
    print("Enter valid input")
else:
    perform_all_op(w)

