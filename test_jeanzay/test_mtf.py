import horovod.tensorflow as hvd
import tensorflow as tf
import time

hvd.init()

try:
    hvd.nccl_create_process_groups([[1,2],[2,3,0]])
except:
    print("\nPYTHON : Expected exception as hvd.nccl_create_proces_groups() returns a vector of vectors.\n")

try:
    #print("\nPYTHON : Tensor before alltoall. \n")
    #t = tf.convert_to_tensor(tf.constant([1,2,3,4,5,6]),dtype=tf.int32)
    #t = [1,2,3,4,5,6]
    #t = [[0.1,0.1,0,0,0,0],[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3]]
    t = [[0.1,0.1,0.1,0.1,0.1,0.1],[1.1,1.1,1.1,1.1,1.1,1.1],[2.1,2.1,2.1,2.1,2.1,2.1],[3.3,3.3,3.3,3.3,3.3,3.3]]
    print("\nPYTHON : Tensor before alltoall = ", t)
    #hvd.alltoall(tf.convert_to_tensor(tf.constant([1,2,3,4,5,6]),dtype=tf.int32), process_group=0)
    #time.sleep(5)
    hvd.alltoall(t, process_group=0)
    #hvd.alltoall(t)
    #hvd.alltoall(t, process_group=0)
    time.sleep(1000)
    print("\nPYTHON : Tensor after alltoall.\n")
except:
    print("\nPYTHON : Exception in hvd.alltoall.\n");

