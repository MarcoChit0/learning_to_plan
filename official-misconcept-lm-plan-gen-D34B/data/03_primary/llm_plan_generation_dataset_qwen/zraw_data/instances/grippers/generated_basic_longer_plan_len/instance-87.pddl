(define (problem gripper-1-10-6)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room1 room2 room3 room4 room5 room6 room7 room8 room9 room10 - room
ball1 ball2 ball3 ball4 ball5 ball6 - obj)
(:init
(at-robby robot1 room1)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at ball1 room10)
(at ball2 room4)
(at ball3 room9)
(at ball4 room2)
(at ball5 room4)
(at ball6 room1)
)
(:goal
(and
(at ball1 room1)
(at ball2 room7)
(at ball3 room1)
(at ball4 room9)
(at ball5 room9)
(at ball6 room3)
)
)
)
