# convolution config setting -> (Kernal size, Num of Filters, Strides, Padding)
# maxpooling config setting -> "M" (defualt of 2 strides)
# convolution sequence config setting -> list[(Kernal size, Num of Filters, Strides, Padding),(),repeat_n]

archi_config=[
    (7,64,2,3),
    "M",
    (3,192,1,1),
    "M",
    (1,128,1,0),
    (3,256,1,1),
    (1,256,1,0),
    (3,512,1,1),
    "M",
    [(1,256,1,0),(3,512,1,1),4],
    (1,512,1,0),
    (3,1024,1,1),
    "M",
    [(1,512,1,0),(3,1024,1,1),2],
    (3,1024,1,1),
    (3,1024,2,1),
    (3,1024,1,1),
    (3,1024,1,1),

]