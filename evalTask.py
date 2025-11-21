task1 = [2, 10, 5, 32, 31, 12,      27, 8, 13, 14, 20,      3, 9, 28, 7, 4,         6, 1, 33, 35, 18,   11, 47, 15, 43, 25,     46, 26, 21, 39, 23,     41, 40, 34, 16, 38,     45, 49, 44, 30, 42,     48, 50, 24, 37, 22,     36]
task2 = [27, 2, 10, 5, 31,          14, 6, 12, 9, 3,        8, 32, 11, 20, 13,      4, 15, 7, 28, 43,   35, 40, 33, 1, 47,      46, 39, 18, 34, 23,     38, 16, 41, 25, 49,     26, 21, 44, 45, 30,     42, 50, 37]
task3 = [27, 12, 38, 20, 33,        14, 43, 28, 47, 26,     15, 11, 5, 35, 34,      6, 39, 23]
task4 = [12, 27, 14, 15, 11,        5, 6, 33]
task5 = [12, 14, 15, 11, 5,         6]
task6 = [27, 47, 35, 38, 41]
task7 = [27, 47, 35, 38, 41]
task8 = [35, 47, 37, 38, 39]
task9 = [16, 38, 21]
task10 = [38, 27, 21]

task1001 = [16, 27, 38, 21, 31, 22, 17, 49, 18, 8, 39, 15, 34]
task1002 = [16, 27, 38, 21, 31, 22, 17, 49, 18, 8, 39, 15, 34]
task1003 = [38, 27, 31, 6, 16, 40, 14, 39, 43, 49, 18, 41, 15, 46, 50, 25, 23, 22, 10, 34, 32]
task1004 = [38, 27, 31, 6, 16, 40, 14, 39, 43, 49, 18, 41, 15, 46, 50, 25, 23, 22, 10, 34, 32]
task1005 = [38, 49, 34, 30, 22, 39]

def evalTask(taskId, ans):
    if taskId == 1:
        expect = task1
    elif taskId == 2:
        expect = task2
    elif taskId == 3:
        expect = task3
    elif taskId == 4:
        expect = task4
    elif taskId == 5:
        expect = task5
    elif taskId == 6:
        expect = task6
    elif taskId == 7:
        expect = task7
    elif taskId == 8:
        expect = task8
    elif taskId == 9:
        expect = task9
    elif taskId == 10:
        expect = task10
    elif taskId == 1001:
        expect = task1001
    elif taskId == 1002:
        expect = task1002
    elif taskId == 1003:
        expect = task1003
    elif taskId == 1004:
        expect = task1004
    elif taskId == 1005:
        expect = task1005   
    
    # elif taskId == 11:
    #     expect = task11
    # elif taskId == 12:
    #     expect = task12
    # elif taskId == 13:
    #     expect = task13
    # elif taskId == 14:
    #     expect = task14
    # elif taskId == 15:
    #     expect = task15
    else:
        print("No such query")
        return 0
    meet = 0
    for i in ans:
        if i in expect:
            meet += 1
    return meet

def evalTaskBestFit(taskId, ans):
    if taskId == 1:
        expect = task1
    elif taskId == 2:
        expect = task2
    elif taskId == 3:
        expect = task3
    elif taskId == 4:
        expect = task4
    elif taskId == 5:
        expect = task5
    elif taskId == 6:
        expect = task6
    elif taskId == 7:
        expect = task7
    elif taskId == 8:
        expect = task8
    elif taskId == 9:
        expect = task9
    elif taskId == 10:
        expect = task10
    elif taskId == 1001:
        expect = task1001
    elif taskId == 1002:
        expect = task1002
    elif taskId == 1003:
        expect = task1003
    elif taskId == 1004:
        expect = task1004
    elif taskId == 1005:
        expect = task1005
    # elif taskId == 11:
    #     expect = task11
    # elif taskId == 12:
    #     expect = task12
    # elif taskId == 13:
    #     expect = task13
    # elif taskId == 14:
    #     expect = task14
    # elif taskId == 15:
    #     expect = task15
    else:
        print("No such query")
        return -1
    for i in ans:
        if i in expect:
            return i
    return -1
