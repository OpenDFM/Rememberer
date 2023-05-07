clean_1_0 = """```
Task:
put a clean lettuce in diningtable.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:

Observations:

---

Encouraged:
go to fridge 1 -> 1.0 First I need to find a lettuce. A lettuce is more likely to appear in fridge (1), diningtable (1), sinkbasin (1), stoveburner (1-3), cabinet (1-13). I can check one by one, starting with fridge 1.
Discouraged:
go to coffeemachine 1 -> 0.0 The lettuce shouldn't be in a coffeemachine.
"""

clean_1_1 = """```
Task:
put a clean lettuce in diningtable.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:
go to fridge 1 First I need to find a lettuce. A lettuce is more likely to appear in fridge (1), diningtable (1), sinkbasin (1), stoveburner (1-3), cabinet (1-13). I can check one by one, starting with fridge 1.
open fridge 1 I need to open fridge 1 to check if there is a lettuce.
go to diningtable 1 There is no lettuce in fridge 1. I can check diningtable 1 then.
take lettuce 1 from diningtable 1 Now I find a lettuce (1). Next, I need to take it.
go to sinkbasin 1 Now I take a lettuce (1). Next, I need to go to sinkbasin (1) and clean it.
Observations:
The fridge 1 is closed.
You open the fridge 1. The fridge 1 is open. In it, you see a cup 3, a egg 2, a potato 3, and a potato 2.
On the diningtable 1, you see a apple 1, a bread 1, a butterknife 2, a cup 2, a fork 2, a knife 2, a knife 1, a ladle 1, a lettuce 1, a mug 2, a mug 1, a pan 2, a peppershaker 1, a spatula 3, a tomato 2, and a tomato 1.
You pick up the lettuce 1 from the diningtable 1.
On the sinkbasin 1, you see a apple 2, a ladle 2, a spoon 1, and a tomato 3.
---

Encouraged:
clean lettuce 1 with sinkbasin 1 -> 1.0 I arrived at sinkbasin 1. Next, I need to clean lettuce 1 with it.
Discouraged:
clean lettuce 1 with stoveburner 1 -> 0.0 Stoveburner cannot be used for cleaning.
"""

clean_1_2 = """```
Task:
put a clean lettuce in diningtable.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:
go to fridge 1 First I need to find a lettuce. A lettuce is more likely to appear in fridge (1), diningtable (1), sinkbasin (1), stoveburner (1-3), cabinet (1-13). I can check one by one, starting with fridge 1.
open fridge 1 I need to open fridge 1 to check if there is a lettuce.
go to diningtable 1 There is no lettuce in fridge 1. I can check diningtable 1 then.
take lettuce 1 from diningtable 1 Now I find a lettuce (1). Next, I need to take it.
go to sinkbasin 1 Now I take a lettuce (1). Next, I need to go to sinkbasin (1) and clean it.
clean lettuce 1 with sinkbasin 1 I arrived at sinkbasin 1. Next, I need to clean lettuce 1 with it.
go to diningtable 1 Now I clean a lettuce (1). Next, I need to go to diningtable 1 to put lettuce 1 in/on it.
Observations:
The fridge 1 is closed.
You open the fridge 1. The fridge 1 is open. In it, you see a cup 3, a egg 2, a potato 3, and a potato 2.
On the diningtable 1, you see a apple 1, a bread 1, a butterknife 2, a cup 2, a fork 2, a knife 2, a knife 1, a ladle 1, a lettuce 1, a mug 2, a mug 1, a pan 2, a peppershaker 1, a spatula 3, a tomato 2, and a tomato 1.
You pick up the lettuce 1 from the diningtable 1.
On the sinkbasin 1, you see a apple 2, a ladle 2, a spoon 1, and a tomato 3.
You clean the lettuce 1 using the sinkbasin 1.
On the diningtable 1, you see a apple 1, a bread 1, a butterknife 2, a cup 2, a fork 2, a knife 2, a knife 1, a ladle 1, a mug 2, a mug 1, a pan 2, a peppershaker 1, a spatula 3, a tomato 2, and a tomato 1.
---

Encouraged:
put lettuce 1 in/on diningtable 1 -> 1.0 I arrived at diningtable 1. Next, I need to put the lettuce in/on it.
Discouraged:
go to diningtable 2 -> 0.0 I shouldn't go to other places.
"""

clean_2_0 = """```
Task:
clean some apple and put it in sidetable.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 3, a diningtable 2, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sidetable 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:

Observations:

---

Encouraged:
go to fridge 1 -> 1.0 First I need to find an apple. An apple is more likely to appear in fridges (1), diningtable (1-3), sidetable (1), countertop (1), sinkbasin (1), garbagecan (1). I can check one by one, starting with fridge 1.
Discouraged:
go to coffeemachine 1 -> 0.0 The lettuce shouldn't be in a coffeemachine.
"""

clean_2_1 = """```
Task:
clean some apple and put it in sidetable.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 3, a diningtable 2, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sidetable 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:
go to fridge 1 First I need to find an apple. An apple is more likely to appear in fridges (1), diningtable (1-3), sidetable (1), countertop (1), sinkbasin (1), garbagecan (1). I can check one by one, starting with fridge 1.
open fridge 1 I need to open fridge 1 to check if there is an apple.
go to diningtable 1 There is no apple in fridge 1. I can check diningtable 1 then.
go to diningtable 2 There is no apple on diningtable 1. I can check diningtable 2 then.
go to diningtable 3 There is no apple on diningtable 2. I can check diningtable 3 then.
go to sidetable 1 There is no apple on diningtable 3. I can check sidetable 1 then.
go to countertop 1 There is no apple on sidetable 1. I can check countertop 1 then.
go to sinkbasin 1 There is no apple on countertop 1. I can check sinkbasin 1 then.
go to garbagecan 1 There is no apple in sinkbasin 1. I can check garbagecan 1 then.
take apple 3 from garbagecan 1 Now I find an apple (3). Next, I need to take it.
go to sinkbasin 1 Now I take an apple (3). Next, I need to go to sinkbasin (1) and clean it.
Observations:
The fridge 1 is closed.
You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 2, a mug 2, a potato 2, and a tomato 1.
On the diningtable 1, you see nothing.
On the diningtable 2, you see a pineapple 1.
On the diningtable 3, you see a bowl 2, a bowl 1, a bread 2, and a bread 1.
On the sidetable 1, you see a cup 1, a lettuce 1, a peppershaker 3, a potato 1, and a saltshaker 1.
On the countertop 1, you see a butterknife 1, a dishsponge 1, and a soapbottle 1.
On the sinkbasin 1, you see a cup 2, a egg 1, a mug 1, and a potato 3.
On the garbagecan 1, you see a apple 3, and a egg 3.
You pick up the apple 3 from the garbagecan 1.
On the sinkbasin 1, you see a cup 2, a egg 1, a mug 1, and a potato 3.
---

Encouraged:
clean apple 3 with sinkbasin 1 -> 1.0 I arrived at sinkbasin 1. Next, I need to clean apple 3 with it.
Discouraged:
heat apple 3 with toaster 1 -> 0.0 I need to clean the apple but not to heat it.
"""


clean_2_2 = """```
Task:
clean some apple and put it in sidetable.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 3, a diningtable 2, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sidetable 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:
go to fridge 1 First I need to find an apple. An apple is more likely to appear in fridges (1), diningtable (1-3), sidetable (1), countertop (1), sinkbasin (1), garbagecan (1). I can check one by one, starting with fridge 1.
open fridge 1 I need to open fridge 1 to check if there is an apple.
go to diningtable 1 There is no apple in fridge 1. I can check diningtable 1 then.
go to diningtable 2 There is no apple on diningtable 1. I can check diningtable 2 then.
go to diningtable 3 There is no apple on diningtable 2. I can check diningtable 3 then.
go to sidetable 1 There is no apple on diningtable 3. I can check sidetable 1 then.
go to countertop 1 There is no apple on sidetable 1. I can check countertop 1 then.
go to sinkbasin 1 There is no apple on countertop 1. I can check sinkbasin 1 then.
go to garbagecan 1 There is no apple in sinkbasin 1. I can check garbagecan 1 then.
take apple 3 from garbagecan 1 Now I find an apple (3). Next, I need to take it.
go to sinkbasin 1 Now I take an apple (3). Next, I need to go to sinkbasin (1) and clean it.
clean apple 3 with sinkbasin 1 I arrived at sinkbasin 1. Next, I need to clean apple 3 with it.
go to sidetable 1 Now I clean an apple (3). Next, I need to go to sidetable 1 and put apple 3 in/on it.
Observations:
The fridge 1 is closed.
You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 2, a mug 2, a potato 2, and a tomato 1.
On the diningtable 1, you see nothing.
On the diningtable 2, you see a pineapple 1.
On the diningtable 3, you see a bowl 2, a bowl 1, a bread 2, and a bread 1.
On the sidetable 1, you see a cup 1, a lettuce 1, a peppershaker 3, a potato 1, and a saltshaker 1.
On the countertop 1, you see a butterknife 1, a dishsponge 1, and a soapbottle 1.
On the sinkbasin 1, you see a cup 2, a egg 1, a mug 1, and a potato 3.
On the garbagecan 1, you see a apple 3, and a egg 3.
You pick up the apple 3 from the garbagecan 1.
On the sinkbasin 1, you see a cup 2, a egg 1, a mug 1, and a potato 3.
You clean the apple 3 using the sinkbasin 1.
On the sidetable 1, you see a cup 1, a lettuce 1, a peppershaker 3, a potato 1, and a saltshaker 1.
---

Encouraged:
put apple 3 in/on sidetable 1 -> 1.0 I arrived at sidetable 1. Next, I need to put apple 3 in/on it.
Discouraged:
go to garbagecan 1 -> 0.0 I shouldn't go to other places.
"""

cool_1_0 = """```
Task:
cool some pan and put it in stoveburner.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 16, a cabinet 15, a cabinet 14, a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a safe 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:

Observations:

---

Encouraged:
go to stoveburner 1 -> 1.0 First I need to find a pan. An pan is more likely to appear in stoveburner (1-4), sinkbasin (1), diningtable (1), countertop (1-2), cabinet (1-16), drawer (1-5). I can check one by one, starting with stoveburner 1.
Discouraged:
go to garbagecan 1 -> 0.0 The pan shouldn't be in a garbagecan.
"""

cool_1_1 = """```
Task:
cool some pan and put it in stoveburner.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 16, a cabinet 15, a cabinet 14, a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a safe 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:
go to stoveburner 1 First I need to find a pan. An pan is more likely to appear in stoveburner (1-4), sinkbasin (1), diningtable (1), countertop (1-2), cabinet (1-16), drawer (1-5). I can check one by one, starting with stoveburner 1.
go to stoveburner 2 There is no pan in stoveburner 1. I can check stoveburner 2 then.
go to stoveburner 3 There is no pan in stoveburner 2. I can check stoveburner 3 then.
take pan 1 from stoveburner 3 Now I find a pan (1). Next, I need to take it.
go to fridge 1 Now I take a pan (1). Next, I need to go to a fridge (1) and cool it.
Observations:
On the stoveburner 1, you see nothing.
On the stoveburner 2, you see a pot 1.
On the stoveburner 3, you see a pan 1.
You pick up the pan 1 from the stoveburner 3.
The fridge 1 is closed.
---

Encouraged:
cool pan 1 with fridge 1 -> 1.0 I arrived at fridge 1. Next, I need to cool pan 1 with it.
Discouraged:
heat pan 1 with fridge 1 -> 0.0 I need to cool the pan but not to heat it.
"""

cool_1_2 = """```
Task:
cool some pan and put it in stoveburner.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 16, a cabinet 15, a cabinet 14, a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a safe 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:
go to stoveburner 1 First I need to find a pan. An pan is more likely to appear in stoveburner (1-4), sinkbasin (1), diningtable (1), countertop (1-2), cabinet (1-16), drawer (1-5). I can check one by one, starting with stoveburner 1.
go to stoveburner 2 There is no pan in stoveburner 1. I can check stoveburner 2 then.
go to stoveburner 3 There is no pan in stoveburner 2. I can check stoveburner 3 then.
take pan 1 from stoveburner 3 Now I find a pan (1). Next, I need to take it.
go to fridge 1 Now I take a pan (1). Next, I need to go to a fridge (1) and cool it.
cool pan 1 with fridge 1 I arrived at fridge 1. Next, I need to cool pan 1 with it.
go to stoveburner 1 Now I cool a pan (1). Next, I need to go to stoveburner 1 to put pan 1 in/on it.
Observations:
On the stoveburner 1, you see nothing.
On the stoveburner 2, you see a pot 1.
On the stoveburner 3, you see a pan 1.
You pick up the pan 1 from the stoveburner 3.
The fridge 1 is closed.
You cool the pan 1 using the fridge 1.
On the stoveburner 1, you see nothing.
---

Encouraged:
put pan 1 in/on stoveburner 1 -> 1.0 I arrived at stoveburner 1. Next, I need to put pan 1 in/on it.
Discouraged:
go to drawer 4 -> 0.0 I shouldn't go to other places.
"""

cool_2_0 = """```
Task:
cool some potato and put it in diningtable.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:

Observations:

---

Encouraged:
go to fridge 1 -> 1.0 First I need to find a potato. A potato is more likely to appear in fridge (1), diningtable (1), countertop (1-2), cabinet (1-12), drawer (1-3), garbagecan (1). I can check one by one, starting with fridge 1.
Discouraged:
go to coffeemachine 1 -> 0.0 The potato shouldn't be in a coffeemachine.
"""

cool_2_1 = """```
Task:
cool some potato and put it in diningtable.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:
go to fridge 1 First I need to find a potato. A potato is more likely to appear in fridge (1), diningtable (1), countertop (1-2), cabinet (1-12), drawer (1-3), garbagecan (1). I can check one by one, starting with fridge 1.
open fridge 1 I need to open fridge 1 to check if there is a potato.
go to diningtable 1 There is no potato in fridge 1. I can check diningtable 1 then.
take potato 2 from diningtable 1 Now I find a potato (2). Next, I need to take it.
go to fridge 1 Now I take a potato (2). Next, I need to go to a fridge (1) and cool it.
Observations:
The fridge 1 is closed.
You open the fridge 1. The fridge 1 is open. In it, you see a apple 1, a bowl 1, a lettuce 2, and a lettuce 1.
On the diningtable 1, you see a apple 3, a bread 1, a butterknife 3, a egg 2, a fork 1, a knife 2, a knife 1, a mug 1, a peppershaker 3, a plate 1, a potato 4, a potato 2, a saltshaker 2, a soapbottle 2, and a tomato 3.
You pick up the potato 2 from the diningtable 1.
The fridge 1 is open. In it, you see a apple 1, a bowl 1, a lettuce 2, and a lettuce 1.
---

Encouraged:
cool potato 2 with fridge 1 -> 1.0 I arrived at fridge 1. Next, I need to cool potato 2 with it.
Discouraged:
cool potato 2 with cabinet 9 -> 0.0 Cabinet cannot be used for cooling.
"""

cool_2_2 = """```
Task:
cool some potato and put it in diningtable.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:
go to fridge 1 First I need to find a potato. A potato is more likely to appear in fridge (1), diningtable (1), countertop (1-2), cabinet (1-12), drawer (1-3), garbagecan (1). I can check one by one, starting with fridge 1.
open fridge 1 I need to open fridge 1 to check if there is a potato.
go to diningtable 1 There is no potato in fridge 1. I can check diningtable 1 then.
take potato 2 from diningtable 1 Now I find a potato (2). Next, I need to take it.
go to fridge 1 Now I take a potato (2). Next, I need to go to a fridge (1) and cool it.
cool potato 2 with fridge 1 I arrived at fridge 1. Next, I need to cool potato 2 with it.
go to diningtable 1 Now I cool a potato (2). Next, I need to go to diningtable 1 to put potato 2 in/on it.
Observations:
The fridge 1 is closed.
You open the fridge 1. The fridge 1 is open. In it, you see a apple 1, a bowl 1, a lettuce 2, and a lettuce 1.
On the diningtable 1, you see a apple 3, a bread 1, a butterknife 3, a egg 2, a fork 1, a knife 2, a knife 1, a mug 1, a peppershaker 3, a plate 1, a potato 4, a potato 2, a saltshaker 2, a soapbottle 2, and a tomato 3.
You pick up the potato 2 from the diningtable 1.
The fridge 1 is open. In it, you see a apple 1, a bowl 1, a lettuce 2, and a lettuce 1.
You cool the potato 2 using the fridge 1.
On the diningtable 1, you see a apple 3, a bread 1, a butterknife 3, a egg 2, a fork 1, a knife 2, a knife 1, a mug 1, a peppershaker 3, a plate 1, a potato 4, a saltshaker 2, a soapbottle 2, and a tomato 3.
---

Encouraged:
put potato 2 in/on diningtable 1 -> 1.0 I arrived at diningtable 1. Next, I need to put potato 2 in/on it.
Discouraged:
go to countertop 2 -> 0.0 I shouldn't go to other places.
"""

examine_1_0 = """```
Task:
look at bowl under the desklamp.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a safe 1, a shelf 5, a shelf 4, a shelf 3, a shelf 2, a shelf 1, a sidetable 2, and a sidetable 1.
Action History:

Observations:

---

Encouraged:
go to drawer 1 -> 1.0 First I need to find a bowl. A bowl is more likely to appear in drawer (1-3), desk (1), sidetable (1-2), shelf (1-5), garbagecan (1). I can check one by one, starting with drawer 1.
Discouraged:
go to safe 1 -> 0.0 The bowl shouldn't be in a safe.
"""

examine_1_1 = """```
Task:
look at bowl under the desklamp.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a safe 1, a shelf 5, a shelf 4, a shelf 3, a shelf 2, a shelf 1, a sidetable 2, and a sidetable 1.
Action History:
go to drawer 1 First I need to find a bowl. A bowl is more likely to appear in drawer (1-3), desk (1), sidetable (1-2), shelf (1-5), garbagecan (1). I can check one by one, starting with drawer 1.
go to drawer 2 There is no bowl in drawer 1. I can check drawer 2 then.
open drawer 2 I need to open drawer 2 to check if there is a bowl.
go to drawer 3 There is no bowl in drawer 2. I can check drawer 3 then.
open drawer 3 I need to open drawer 3 to check if there is a bowl.
go to desk 1 There is no bowl in drawer 3. I can check desk 1 then.
go to sidetable 1 There is no bowl on desk 1. I can check sidetable 1 then.
go to sidetable 2 There is no bowl on sidee 1. I can check sidetable 2 then.
go to shelf 1 There is no bowl on sidetable 2. I can check shelf 1 then.
go to shelf 2 There is no bowl on shelf 1. I can check shelf 2 then.
take bowl 1 from shelf 2 Now I find a bowl (1). Next, I need to take it.
Observations:
On the drawer 1, you see nothing.
The drawer 2 is closed.
You open the drawer 2. The drawer 2 is open. In it, you see nothing.
The drawer 3 is closed.
You open the drawer 3. The drawer 3 is open. In it, you see nothing.
On the desk 1, you see a alarmclock 2, a book 1, a cellphone 1, a keychain 1, a laptop 2, a laptop 1, and a pen 3.
On the sidetable 1, you see a cd 1, a pen 1, and a pencil 1.
On the sidetable 2, you see a alarmclock 1, a desklamp 1, and a pen 2.
On the shelf 1, you see nothing.
On the shelf 2, you see a bowl 1.
You pick up the bowl 1 from the shelf 2.
---

Encouraged:
go to sidetable 2 -> 1.0 Now I take a bowl (1). Next, I need to find a desklamp. Desklamp 1 is on sidetable 2. Thus, I can directly go to sidetable 2.
Discouraged:
go to sidetable 1 -> 0.0 There is no desklamp on sidetable 1.
"""

examine_1_2 = """```
Task:
look at bowl under the desklamp.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a safe 1, a shelf 5, a shelf 4, a shelf 3, a shelf 2, a shelf 1, a sidetable 2, and a sidetable 1.
Action History:
go to drawer 1 First I need to find a bowl. A bowl is more likely to appear in drawer (1-3), desk (1), sidetable (1-2), shelf (1-5), garbagecan (1). I can check one by one, starting with drawer 1.
go to drawer 2 There is no bowl in drawer 1. I can check drawer 2 then.
open drawer 2 I need to open drawer 2 to check if there is a bowl.
go to drawer 3 There is no bowl in drawer 2. I can check drawer 3 then.
open drawer 3 I need to open drawer 3 to check if there is a bowl.
go to desk 1 There is no bowl in drawer 3. I can check desk 1 then.
go to sidetable 1 There is no bowl on desk 1. I can check sidetable 1 then.
go to sidetable 2 There is no bowl on sidee 1. I can check sidetable 2 then.
go to shelf 1 There is no bowl on sidetable 2. I can check shelf 1 then.
go to shelf 2 There is no bowl on shelf 1. I can check shelf 2 then.
take bowl 1 from shelf 2 Now I find a bowl (1). Next, I need to take it.
go to sidetable 2 Now I take a bowl (1). Next, I need to find a desklamp. Desklamp 1 is on sidetable 2.
Observations:
On the drawer 1, you see nothing.
The drawer 2 is closed.
You open the drawer 2. The drawer 2 is open. In it, you see nothing.
The drawer 3 is closed.
You open the drawer 3. The drawer 3 is open. In it, you see nothing.
On the desk 1, you see a alarmclock 2, a book 1, a cellphone 1, a keychain 1, a laptop 2, a laptop 1, and a pen 3.
On the sidetable 1, you see a cd 1, a pen 1, and a pencil 1.
On the sidetable 2, you see a alarmclock 1, a desklamp 1, and a pen 2.
On the shelf 1, you see nothing.
On the shelf 2, you see a bowl 1.
You pick up the bowl 1 from the shelf 2.
On the sidetable 2, you see a alarmclock 1, a desklamp 1, and a pen 2.
---

Encouraged:
use desklamp 1 -> 1.0 Now I find a desklamp (1). Next, I need to use it.
Discouraged:
go to sidetable 1 -> 0.0 I shouldn't go to other places.
"""

examine_2_0 = """```
Task:
examine the pen with the desklamp.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a bed 1, a drawer 10, a drawer 9, a drawer 8, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a shelf 9, a shelf 8, a shelf 7, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.
Action History:

Observations:

---

Encouraged:
go to drawer 1 -> 1.0 To solve the task, I need to find and take a pen, then find and use a desklamp. First I need to find a pen. A pen is more likely to appear in drawer (1-10), shelf (1-9), bed (1), garbagecan (1). I can check one by one, starting with drawer 1.
Discouraged:
go to dresser 1 -> 0.0 The pen shouldn't be on a dresser.
"""

examine_2_1 = """```
Task:
examine the pen with the desklamp.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a bed 1, a drawer 10, a drawer 9, a drawer 8, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a shelf 9, a shelf 8, a shelf 7, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.
Action History:
go to drawer 1 To solve the task, I need to find and take a pen, then find and use a desklamp. First I need to find a pen. A pen is more likely to appear in drawer (1-10), shelf (1-9), bed (1), garbagecan (1). I can check one by one, starting with drawer 1.
open drawer 1 I need to open drawer 1 to check if there is a pen.
go to drawer 2 There is no pen in drawer 1. I can check drawer 2 then.
open drawer 2 I need to open drawer 2 to check if there is a pen.
go to drawer 3 There is no pen in drawer 2. I can check drawer 3 then.
open drawer 3 I need to open drawer 3 to check if there is a pen.
go to drawer 4 There is no pen in drawer 3. I can check drawer 4 then.
open drawer 4 I need to open drawer 4 to check if there is a pen.
take pen 2 from drawer 4 Now I find a pen (2). Next, I need to take it.
Observations:
The drawer 1 is closed.
You open the drawer 1. The drawer 1 is open. In it, you see nothing.
The drawer 2 is closed.
You open the drawer 2. The drawer 2 is open. In it, you see nothing.
The drawer 3 is closed.
You open the drawer 3. The drawer 3 is open. In it, you see nothing.
The drawer 4 is closed.
You open the drawer 4. The drawer 4 is open. In it, you see a cd 3, a keychain 1, and a pen 2.
You pick up the pen 2 from the drawer 4.
---

Encouraged:
go to dresser 1 -> 1.0 Now I take a pen (2). Next, I need to find a desklamp. A desklamp is more likely to appear in dresser (1), shelf (1-9), bed (1), garbagecan (1), drawer (1-10). I can check one by one, starting with dresser 1.
Discouraged:
go to drawer 1 -> 0.0 There is no desklamp in drawer 1.
"""

examine_2_2 = """```
Task:
examine the pen with the desklamp.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a bed 1, a drawer 10, a drawer 9, a drawer 8, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a shelf 9, a shelf 8, a shelf 7, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.
Action History:
go to drawer 1 To solve the task, I need to find and take a pen, then find and use a desklamp. First I need to find a pen. A pen is more likely to appear in drawer (1-10), shelf (1-9), bed (1), garbagecan (1). I can check one by one, starting with drawer 1.
open drawer 1 I need to open drawer 1 to check if there is a pen.
go to drawer 2 There is no pen in drawer 1. I can check drawer 2 then.
open drawer 2 I need to open drawer 2 to check if there is a pen.
go to drawer 3 There is no pen in drawer 2. I can check drawer 3 then.
open drawer 3 I need to open drawer 3 to check if there is a pen.
go to drawer 4 There is no pen in drawer 3. I can check drawer 4 then.
open drawer 4 I need to open drawer 4 to check if there is a pen.
take pen 2 from drawer 4 Now I find a pen (2). Next, I need to take it.
go to dresser 1 Now I take a pen (2). Next, I need to find a desklamp. A desklamp is more likely to appear in dresser (1), shelf (1-9), bed (1), garbagecan (1), drawer (1-10). I can check one by one, starting with dresser 1.
Observations:
The drawer 1 is closed.
You open the drawer 1. The drawer 1 is open. In it, you see nothing.
The drawer 2 is closed.
You open the drawer 2. The drawer 2 is open. In it, you see nothing.
The drawer 3 is closed.
You open the drawer 3. The drawer 3 is open. In it, you see nothing.
The drawer 4 is closed.
You open the drawer 4. The drawer 4 is open. In it, you see a cd 3, a keychain 1, and a pen 2.
You pick up the pen 2 from the drawer 4.
On the dresser 1, you see a alarmclock 2, a alarmclock 1, a book 1, a cd 1, a creditcard 1, a desklamp 1, a keychain 2, a pen 3, and a pen 1.
---

Encouraged:
use desklamp 1 -> 1.0 Now I find a desklamp (1). Next, I need to use it.
Discouraged:
go to shelf 1 -> 0.0 I shouldn't go to other places.
"""

heat_1_0 = """```
Task:
heat some egg and put it in diningtable.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a diningtable 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:

Observations:

---

Encouraged:
go to fridge 1 -> 1.0 First I need to find an egg. An egg is more likely to appear in fridge (1), countertop (1-3), diningtable (1), stoveburner (1-4), toaster (1), garbagecan (1), cabinet (1-10). I can check one by one, starting with fridge 1.
Discouraged:
go to coffeemachine 1 -> 0.0 The egg shouldn't be in a coffeemachine.
"""

heat_1_1 = """```
Task:
heat some egg and put it in diningtable.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a diningtable 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:
go to fridge 1 First I need to find an egg. An egg is more likely to appear in fridge (1), countertop (1-3), diningtable (1), stoveburner (1-4), toaster (1), garbagecan (1), cabinet (1-10). I can check one by one, starting with fridge 1.
open fridge 1 I need to open fridge 1 to check if there is an agg.
go to countertop 1 There is no egg in fridge 1. I can check countertop 1 then.
go to countertop 2 There is no egg on countertop 1. I can check countertop 2 then.
go to countertop 3 There is no egg on countertop 2. I can check countertop 3 then.
take egg 2 from countertop 3 Now I find an egg (2). Next, I need to take it.
go to microwave 1 Now I take an egg (2). Next, I need go to a microwave (1) and heat it.
Observations:
The fridge 1 is closed.
You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 2, a mug 2, and a potato 3.
On the countertop 1, you see a bread 1, a fork 1, and a saltshaker 1.
On the countertop 2, you see nothing.
On the countertop 3, you see a bowl 1, a butterknife 1, a egg 2, a kettle 2, a plate 1, a sink 1, and a spatula 2.
You pick up the egg 2 from the countertop 3.
The microwave 1 is closed.
---

Encouraged:
heat egg 2 with microwave 1 -> 1.0 I arrive at microwave 1. Next, I need to heat egg 2 with it.
Discouraged:
heat egg 2 with stoveburner 2 -> 0.0 I'm not aside with stoveburner 2
"""

heat_1_2 = """```
Task:
heat some egg and put it in diningtable.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a diningtable 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:
go to fridge 1 First I need to find an egg. An egg is more likely to appear in fridge (1), countertop (1-3), diningtable (1), stoveburner (1-4), toaster (1), garbagecan (1), cabinet (1-10). I can check one by one, starting with fridge 1.
open fridge 1 I need to open fridge 1 to check if there is an agg.
go to countertop 1 There is no egg in fridge 1. I can check countertop 1 then.
go to countertop 2 There is no egg on countertop 1. I can check countertop 2 then.
go to countertop 3 There is no egg on countertop 2. I can check countertop 3 then.
take egg 2 from countertop 3 Now I find an egg (2). Next, I need to take it.
go to microwave 1 Now I take an egg (2). Next, I need go to a microwave (1) and heat it.
heat egg 2 with microwave 1 I arrive at microwave 1. Next, I need to heat egg 2 with it.
go to diningtable 1 Now I heat an egg (2). Next, I need to go to diningtable 1 to put egg 2 in/on 1.
Observations:
The fridge 1 is closed.
You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 2, a mug 2, and a potato 3.
On the countertop 1, you see a bread 1, a fork 1, and a saltshaker 1.
On the countertop 2, you see nothing.
On the countertop 3, you see a bowl 1, a butterknife 1, a egg 2, a kettle 2, a plate 1, a sink 1, and a spatula 2.
You pick up the egg 2 from the countertop 3.
The microwave 1 is closed.
You heat the egg 2 using the microwave 1.
On the diningtable 1, you see a apple 2, a bread 3, a egg 1, a kettle 1, a knife 1, a mug 1, a papertowelroll 1, a peppershaker 2, a potato 1, a soapbottle 1, and a spatula 1.
---

Encouraged:
put egg 2 in/on diningtable 1 -> 1.0 I arrived at diningtable 1. Next, I need to put egg 2 in/on it.
Discouraged:
go to coutertop 3 -> 0.0 I shouldn't go to other places.
"""

heat_2_0 = """```
Task:
put a hot apple in fridge.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:

Observations:

---

Encouraged:
go to fridge 1 -> 1.0 First I need to find an apple. An apple is more likely to appear in fridge (1), diningtable (1), coffeetable (1), drawer (1), cabinet (1-13), garbagecan (1). I can check one by one, starting with fridge 1.
Discouraged:
go to coffeemachine -> 0.0 The apple shouldn't be in a coffeemachine.
"""

heat_2_1 = """```
Task:
put a hot apple in fridge.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:
go to fridge 1 First I need to find an apple. An apple is more likely to appear in fridge (1), diningtable (1), coffeetable (1), drawer (1), cabinet (1-13), garbagecan (1). I can check one by one, starting with fridge 1.
open fridge 1 I need to open fridge 1 to check if there is an apple.
go to diningtable 1 There is no apple in fridge 1. I can check diningtable then.
take apple 1 from diningtable 1 Now I find an apple (1). Next, I need to take it.
go to microwave 1 Now I take an apple (1). Next, I need to go to a microwave (1) and heat it.
Observations:
The fridge 1 is closed.
You open the fridge 1. The fridge 1 is open. In it, you see a cup 1, and a egg 1.
On the diningtable 1, you see a apple 1, a bread 1, a fork 3, a fork 2, a fork 1, a mug 2, a peppershaker 3, a plate 2, a pot 1, a soapbottle 3, a spatula 1, a spoon 3, a spoon 2, a spoon 1, a winebottle 3, and a winebottle 2.
You pick up the apple 1 from the diningtable 1.
The microwave 1 is closed.
---

Encouraged:
heat apple 1 with microwave 1 -> 1.0 I arrived at microwave 1. Next, I need to heat apple 1 with it.
Discouraged:
clean apple 1 with cabinet 9 -> 0.0 Cabinet cannot be used for heating.
"""

heat_2_2 = """```
Task:
put a hot apple in fridge.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:
go to fridge 1 First I need to find an apple. An apple is more likely to appear in fridge (1), diningtable (1), coffeetable (1), drawer (1), cabinet (1-13), garbagecan (1). I can check one by one, starting with fridge 1.
open fridge 1 I need to open fridge 1 to check if there is an apple.
go to diningtable 1 There is no apple in fridge 1. I can check diningtable then.
take apple 1 from diningtable 1 Now I find an apple (1). Next, I need to take it.
go to microwave 1 Now I take an apple (1). Next, I need to go to a microwave (1) and heat it.
heat apple 1 with microwave 1 -> 1.0 I arrived at microwave 1. Next, I need to heat apple 1 with it.
go to fridge 1 Now I heat an apple (1). Next, I need to go to fridge 1 to put it in/on it.
Observations:
The fridge 1 is closed.
You open the fridge 1. The fridge 1 is open. In it, you see a cup 1, and a egg 1.
On the diningtable 1, you see a apple 1, a bread 1, a fork 3, a fork 2, a fork 1, a mug 2, a peppershaker 3, a plate 2, a pot 1, a soapbottle 3, a spatula 1, a spoon 3, a spoon 2, a spoon 1, a winebottle 3, and a winebottle 2.
You pick up the apple 1 from the diningtable 1.
The microwave 1 is closed.
You heat the apple 1 using the microwave 1.
The fridge 1 is open. In it, you see a cup 1, and a egg 1.
---

Encouraged:
put apple 1 in/on fridge 1 -> 1.0 I arrived at fridge 1. Next, I need to put apple 1 in/on it.
Discouraged:
go to cabinet 11 -> 0.0 I shouldn't go to other places.
"""

put_1_0 = """```
Task:
put some spraybottle on toilet.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Action History:

Observations:

---

Encouraged:
go to cabinet 1 -> 1.0 First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet (1-4), countertop (1), toilet (1), sinkbasin (1-2), garbagecan (1). I can check one by one, starting with cabinet 1.
Discouraged:
go to toiletpaperhanger 1 -> 0.0 The spraybottle shouldn't be on a toiletpaperhanger.
"""

put_1_1 = """```
Task:
put some spraybottle on toilet.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Action History:
go to cabinet 1 First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet (1-4), countertop (1), toilet (1), sinkbasin (1-2), garbagecan (1). I can check one by one, starting with cabinet 1.
go to cabinet 2 There is no spraybottle on cabinet 1. I can check cabinet 2 then.
open cabinet 2 I need to open cabinet 2 to check if there is a spraybottle.
Observations:
On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.
The cabinet 2 is closed.
You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.
---

Encouraged:
take spraybottle 2 from cabinet 2 -> 1.0 Now I find a spraybottle (2). Next, I need to take it.
Discouraged:
go to cabinet 1 -> 0.0 The spraybottle is in cabinet 2 and I shouldn't go to cabinet 1.
"""

put_1_2 = """```
Task:
put some spraybottle on toilet.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Action History:
go to cabinet 1 First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet (1-4), countertop (1), toilet (1), sinkbasin (1-2), garbagecan (1). I can check one by one, starting with cabinet 1.
go to cabinet 2 There is no spraybottle on cabinet 1. I can check cabinet 2 then.
open cabinet 2 I need to open cabinet 2 to check if there is a spraybottle.
take spraybottle 2 from cabinet 2 Now I find a spraybottle (2). Next, I need to take it.
go to toilet 1 Now I take a spraybottle (2). Next, I need to go to toilet 1 to put spraybottle 2 in/on it.
Observations:
On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.
The cabinet 2 is closed.
You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.
You pick up the spraybottle 2 from the cabinet 2.
On the toilet 1, you see a soapbottle 2.
---

Encouraged:
put spraybottle 2 in/on toilet 1 -> 1.0 Now I arrived at spraybottle 2. Next, I need to put spraybottle 2 in/on it.
Discouraged:
go to sinkbasin 1 -> 0.0 I shouldn't go to other places.
"""

put_2_0 = """```
Task:
find some apple and put it in sidetable.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 3, a diningtable 2, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sidetable 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:

Observations:

---

Encouraged:
go to fridge 1 -> 1.0 First I need to find an apple. An apple is more likely to appear in fridges (1), diningtables (1-3), sidetables (1), countertops (1), sinkbasins (1), garbagecan (1). I can check one by one, starting with fridge 1.
Discouraged:
go to toaster 1 -> 0.0 The apple shouldn't be in a toaster.
"""

put_2_1 = """```
Task:
find some apple and put it in sidetable.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 3, a diningtable 2, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sidetable 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:
go to fridge 1 First I need to find an apple. An apple is more likely to appear in fridges (1), diningtables (1-3), sidetables (1), countertops (1), sinkbasins (1), garbagecan (1). I can check one by one, starting with fridge 1.
open fridge 1 I need to open fridge 1 to check if there is an apple.
go to diningtable 1 There is no apple in fridge 1. I can check diningtable 1 then.
go to diningtable 2 There is no apple on diningtable 1. I can check diningtable 2 then.
go to diningtable 3 There is no apple on diningtable 2. I can check diningtable 3 then.
go to sidetable 1 There is no apple on diningtable 3. I can check sidetable 1 then.
go to countertop 1 There is no apple on sidetable 1. I can check countertop 1 then.
go to sinkbasin 1 There is no apple on countertop 1. I can check sinkbasin 1 then.
go to garbagecan 1 There is no apple in sinkbasin 1. I can check garbagecan 1 then.
Observations:
The fridge 1 is closed.
You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 2, a mug 2, a potato 2, and a tomato 1.
On the diningtable 1, you see nothing.
On the diningtable 2, you see a pineapple 1.
On the diningtable 3, you see a bowl 2, a bowl 1, a bread 2, and a bread 1.
On the sidetable 1, you see a cup 1, a lettuce 1, a peppershaker 3, a potato 1, and a saltshaker 1.
On the countertop 1, you see a butterknife 1, a dishsponge 1, and a soapbottle 1.
On the sinkbasin 1, you see a cup 2, a egg 1, a mug 1, and a potato 3.
On the garbagecan 1, you see a apple 3, and a egg 3.
---

Encouraged:
take apple 3 from garbagecan 1 -> 1.0 Now I find an apple (3). Next, I need to take it.
Discouraged:
take egg 3 from garbagecan 1 -> 0.0 I need to take apple 3 but not egg 3.
"""

put_2_2 = """```
Task:
find some apple and put it in sidetable.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 3, a diningtable 2, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sidetable 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action History:
go to fridge 1 First I need to find an apple. An apple is more likely to appear in fridges (1), diningtables (1-3), sidetables (1), countertops (1), sinkbasins (1), garbagecan (1). I can check one by one, starting with fridge 1.
open fridge 1 I need to open fridge 1 to check if there is an apple.
go to diningtable 1 There is no apple in fridge 1. I can check diningtable 1 then.
go to diningtable 2 There is no apple on diningtable 1. I can check diningtable 2 then.
go to diningtable 3 There is no apple on diningtable 2. I can check diningtable 3 then.
go to sidetable 1 There is no apple on diningtable 3. I can check sidetable 1 then.
go to countertop 1 There is no apple on sidetable 1. I can check countertop 1 then.
go to sinkbasin 1 There is no apple on countertop 1. I can check sinkbasin 1 then.
go to garbagecan 1 There is no apple in sinkbasin 1. I can check garbagecan 1 then.
take apple 3 from garbagecan 1 Now I find an apple (3). Next, I need to take it.
go to sidetable 1 Now I find an apple (3). Next, I need to go to sidetable 1 to put apple 3 in/on it.
Observations:
The fridge 1 is closed.
You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 2, a mug 2, a potato 2, and a tomato 1.
On the diningtable 1, you see nothing.
On the diningtable 2, you see a pineapple 1.
On the diningtable 3, you see a bowl 2, a bowl 1, a bread 2, and a bread 1.
On the sidetable 1, you see a cup 1, a lettuce 1, a peppershaker 3, a potato 1, and a saltshaker 1.
On the countertop 1, you see a butterknife 1, a dishsponge 1, and a soapbottle 1.
On the sinkbasin 1, you see a cup 2, a egg 1, a mug 1, and a potato 3.
On the garbagecan 1, you see a apple 3, and a egg 3.
You pick up the apple 3 from the garbagecan 1.
On the sidetable 1, you see a cup 1, a lettuce 1, a peppershaker 3, a potato 1, and a saltshaker 1.
---

Encouraged:
put apple 3 in/on sidetable 1 -> 1.0 I arrived at sidetable 1. Next, I need to put apple 3 in/on it.
Discouraged:
put egg 3 in/on sidetable 1 -> 0.0 I don't carry egg 3.
"""

puttwo_1_0 = """```
Task:
put two creditcard in dresser.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a armchair 2, a armchair 1, a bed 1, a countertop 1, a diningtable 1, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a laundryhamper 1, and a sidetable 1.
Action History:

Observations:

---

Encouraged:
go to drawer 1 -> 1.0 First I need to find the first creditcard. A creditcard is more likely to appear in drawer (1-2), coutertop (1), sidetable (1), diningtable (1), armchair (1-2), bed (1). I can check one by one, starting with drawer 1.
Discouraged:
go to laundryhamper 1 -> 0.0 The creditcard shouldn't be in a laundryhamper.
"""

puttwo_1_1 = """```
Task:
put two creditcard in dresser.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a armchair 2, a armchair 1, a bed 1, a countertop 1, a diningtable 1, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a laundryhamper 1, and a sidetable 1.
Action History:
go to drawer 1 First I need to find the first creditcard. A creditcard is more likely to appear in drawer (1-2), coutertop (1), sidetable (1), diningtable (1), armchair (1-2), bed (1). I can check one by one, starting with drawer 1.
open drawer 1 I need to open drawer 1 to check if there is a creditcard.
go to drawer 2 There is no creditcard in drawer 1. I can check drawer 2 then.
open drawer 2 I need to open drawer 2 to check if there is a creditcard.
go to countertop 1 There is no creditcard in drawer 2. I can check coutertop 1 then.
take creditcard 2 from countertop 1 Now I find the first creditcard (2). Next, I need to take it. I can find the second creditcard (3) later in countertop 1.
go to dresser 1 Now I take the first creditcard (2). Next, I need to go to dresser 1 to put creditcard 2 in/on it.
Observations:
The drawer 1 is closed.
You open the drawer 1. The drawer 1 is open. In it, you see a book 1, a cd 1, and a pen 1.
The drawer 2 is closed.
You open the drawer 2. The drawer 2 is open. In it, you see nothing.
On the countertop 1, you see a cellphone 2, a creditcard 4, a creditcard 3, a creditcard 2, a mirror 1, a pencil 2, and a pencil 1.
You pick up the creditcard 2 from the countertop 1.
On the dresser 1, you see a mug 1, and a television 1.
---

Encouraged:
put creditcard 2 in/on dresser 1 -> 1.0 I arrived at dresser 1. Next, I need to put creditcard 2 in/on it.
Discouraged:
take mug 1 -> 0.0 I need to put creditcard 2 down but not take other objects.
"""

puttwo_1_2 = """```
Task:
put two creditcard in dresser.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a armchair 2, a armchair 1, a bed 1, a countertop 1, a diningtable 1, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a laundryhamper 1, and a sidetable 1.
Action History:
go to drawer 1 First I need to find the first creditcard. A creditcard is more likely to appear in drawer (1-2), coutertop (1), sidetable (1), diningtable (1), armchair (1-2), bed (1). I can check one by one, starting with drawer 1.
open drawer 1 I need to open drawer 1 to check if there is a creditcard.
go to drawer 2 There is no creditcard in drawer 1. I can check drawer 2 then.
open drawer 2 I need to open drawer 2 to check if there is a creditcard.
go to countertop 1 There is no creditcard in drawer 2. I can check coutertop 1 then.
take creditcard 2 from countertop 1 Now I find the first creditcard (2). Next, I need to take it. I can find the second creditcard (3) later in countertop 1.
go to dresser 1 Now I take the first creditcard (2). Next, I need to go to dresser 1 to put creditcard 2 in/on it.
put creditcard 2 in/on dresser 1 I arrived at dresser 1. Next, I need to put creditcard 2 in/on it.
Observations:
The drawer 1 is closed.
You open the drawer 1. The drawer 1 is open. In it, you see a book 1, a cd 1, and a pen 1.
The drawer 2 is closed.
You open the drawer 2. The drawer 2 is open. In it, you see nothing.
On the countertop 1, you see a cellphone 2, a creditcard 4, a creditcard 3, a creditcard 2, a mirror 1, a pencil 2, and a pencil 1.
You pick up the creditcard 2 from the countertop 1.
On the dresser 1, you see a mug 1, and a television 1.
You put the creditcard 2 in/on the dresser 1.
---

Encouraged:
go to countertop 1 -> 1.0 Now I put the first creditcard in dresser. Next, I need to find the second creditcard. Creditcards (3-4) are on countertop 1. Thus, I can directly go to countertop 1.
Discouraged:
go to drawer 2 -> 0.0 There is no creditcard in drawer 2
"""

puttwo_2_0 = """```
Task:
put two cellphone in sofa.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a coffeetable 1, a diningtable 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a sidetable 2, a sidetable 1, and a sofa 1.
Action History:

Observations:

---

Encouraged:
go to coffeetable 1 -> 1.0 First I need to find the first cellphone. A cellphone is more likely to appear in coffeetable (1), diningtable (1), sidetable (1-2), drawer (1-4), sofa (1), dresser (1), garbagecan (1). I can check one by one, starting with coffeetable 1.
Discouraged:
go to sofa 1 -> 0.0 I need to take cellphone to sofa, but not take it from sofa.
"""

puttwo_2_1 = """```
Task:
put two cellphone in sofa.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a coffeetable 1, a diningtable 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a sidetable 2, a sidetable 1, and a sofa 1.
Action History:
go to coffeetable 1 First I need to find the first cellphone. A cellphone is more likely to appear in coffeetable (1), diningtable (1), sidetable (1-2), drawer (1-4), sofa (1), dresser (1), garbagecan (1). I can check one by one, starting with coffeetable 1.
take cellphone 3 from coffeetable 1 Now I find the first cellphone (3). Next, I need to take it.
go to sofa 1 Now I take the first cellphone (3). Next, I need to go to sofa 1 to put cellphone 3 in/on it.
put cellphone 3 in/on sofa 1 I arrived  at sofa 1. Next, I need put cellphone 3 in/on it.
Observations:
On the coffeetable 1, you see a box 1, a cellphone 3, and a plate 1.
You pick up the cellphone 3 from the coffeetable 1.
On the sofa 1, you see a newspaper 2, a pillow 1, a remotecontrol 2, and a remotecontrol 1.
You put the cellphone 3 in/on the sofa 1.
---

Encouraged:
go to diningtable 1 -> 1.0 Now I put the first cellphone in sofa. Next, I need to find the second cellphone. I have visited coffeetable (1), so the second cellphone is more likely to appear in diningtable (1), sidetable (1-2), drawer (1-4), sofa (1), dresser (1), garbagecan (1). I can check one by one, starting with diningtable 1.
Discouraged:
go to coffeetable 1 -> 0.0 I have visited coffeetable 1. There is no more cellphone on coffeetable 1.
"""

puttwo_2_2 = """```
Task:
put two cellphone in sofa.
Initial State:
You are in the middle of a room. Looking quickly around you, you see a coffeetable 1, a diningtable 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a sidetable 2, a sidetable 1, and a sofa 1.
Action History:
go to coffeetable 1 First I need to find the first cellphone. A cellphone is more likely to appear in coffeetable (1), diningtable (1), sidetable (1-2), drawer (1-4), sofa (1), dresser (1), garbagecan (1). I can check one by one, starting with coffeetable 1.
take cellphone 3 from coffeetable 1 Now I find the first cellphone (3). Next, I need to take it.
go to sofa 1 Now I take the first cellphone (3). Next, I need to go to sofa 1 to put cellphone 3 in/on it.
put cellphone 3 in/on sofa 1 I arrived  at sofa 1. Next, I need put cellphone 3 in/on it.
go to diningtable 1 -> 1.0 Now I put the first cellphone in sofa. Next, I need to find the second cellphone. I have visited coffeetable (1), so the second cellphone is more likely to appear in diningtable (1), sidetable (1-2), drawer (1-4), sofa (1), dresser (1), garbagecan (1). I can check one by one, starting with diningtable 1.
take cellphone 2 from diningtable 1 Now I find the second cellphone (2). Next, I need to take it.
go to sofa 1 Now I take the second cellphone (2). Next, I need to go to sofa 1 to put cellphone 2 in/on it.
Observations:
On the coffeetable 1, you see a box 1, a cellphone 3, and a plate 1.
You pick up the cellphone 3 from the coffeetable 1.
On the sofa 1, you see a newspaper 2, a pillow 1, a remotecontrol 2, and a remotecontrol 1.
You put the cellphone 3 in/on the sofa 1.
On the diningtable 1, you see a cellphone 2, a keychain 2, a laptop 1, a statue 2, and a statue 1.
You pick up the cellphone 2 from the diningtable 1.
On the sofa 1, you see a cellphone 3, a newspaper 2, a pillow 1, a remotecontrol 2, and a remotecontrol 1.
---

Encouraged:
put cellphone 2 in/on sofa 1 -> 1.0 I arrived at sofa 1. Next, I need to put cellphone 2 in/on sofa 1.
Discouraged:
go to sidetable 2 -> 0.0 I shouldn't go to other places.
"""
