(define (problem gripper-1-10-2)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room1 room2 room3 room4 room5 room6 room7 room8 room9 room10 - room
ball1 ball2 - obj)
(:init
(at-robby robot1 room9)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at ball1 room8)
(at ball2 room5)
)
(:goal
(and
(at ball1 room1)
(at ball2 room6)
)
)
)
