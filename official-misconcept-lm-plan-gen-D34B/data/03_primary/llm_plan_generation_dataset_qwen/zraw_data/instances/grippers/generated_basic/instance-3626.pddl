(define (problem gripper-2-10-3)
(:domain gripper-strips)
(:objects robot1 robot2 - robot
rgripper1 lgripper1 rgripper2 lgripper2 - gripper
room1 room2 room3 room4 room5 room6 room7 room8 room9 room10 - room
ball1 ball2 ball3 - obj)
(:init
(at-robby robot1 room7)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room5)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at ball1 room9)
(at ball2 room9)
(at ball3 room7)
)
(:goal
(and
(at ball1 room8)
(at ball2 room6)
(at ball3 room2)
)
)
)
