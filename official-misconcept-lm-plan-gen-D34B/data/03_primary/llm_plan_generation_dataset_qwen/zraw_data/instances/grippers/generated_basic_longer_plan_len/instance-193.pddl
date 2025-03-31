(define (problem gripper-1-9-9)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room1 room2 room3 room4 room5 room6 room7 room8 room9 - room
ball1 ball2 ball3 ball4 ball5 ball6 ball7 ball8 ball9 - obj)
(:init
(at-robby robot1 room3)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at ball1 room9)
(at ball2 room7)
(at ball3 room6)
(at ball4 room4)
(at ball5 room9)
(at ball6 room9)
(at ball7 room8)
(at ball8 room8)
(at ball9 room6)
)
(:goal
(and
(at ball1 room4)
(at ball2 room3)
(at ball3 room4)
(at ball4 room7)
(at ball5 room4)
(at ball6 room9)
(at ball7 room3)
(at ball8 room1)
(at ball9 room7)
)
)
)
