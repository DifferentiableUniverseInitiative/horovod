import boost_nccl as m

b = m.boost_nccl()

# create drow instance
try:
    b.create_process_groups([[0,2],[1 ,2 ,4]])
except:
    print("create_process_groups returns a vector of vectors => expected exception")


