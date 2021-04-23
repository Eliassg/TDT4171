import random

usernames = ["cwapc",
"gvtvj",
"csati",
"ahxmn",
"nhekh",
"dezmt",
"njkfw",
"kmsin",
"flbxt",
"vdlhv",
"gkknh",
"pzibi",
"whprn",
"cuzwi",
"stskm",
"gfcbf",
"dwpzo",
"ojmbx",
"mlnko",
"gcdnc"]

userid = [
    59618,
    84818,
    36687,
    58667,
    30840,
    36669,
    97720,
    48336,
    18462,
    97262,
    23562,
    78062,
    22272,
    28779,
    73040,
    20144,
    47268,
    17787,
    97226,
    94060
]



with open("data.txt", "w") as f: 
    f.write("#Courses")
    for i in range(10):
        course = "\n" + "TDT410" + str(i) + "," + "Example Course " + str(i)
        f.write(course) 

    f.write("\n" + "#Students")
    for i in range(20):
        name = usernames[i-1]
        birtyear = 1990 + random.randint(0, 9)
        user = userid[i-1]
        f.write("\n" + name + "," + str(birtyear) + ',' + str(user))

    f.write("\n" + "#Results")
    for user in userid:
        for i in range(10):
            f.write("\n" + str(user) + "," + str(random.randint(1,5)) + "," + "TDT410" + str(i))

