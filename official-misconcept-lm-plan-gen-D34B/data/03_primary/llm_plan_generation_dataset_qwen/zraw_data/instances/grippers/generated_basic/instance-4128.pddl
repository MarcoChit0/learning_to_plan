(define (problem gripper-2-8-3)
(:domain gripper-strips)
(:objects robot1 robot2 - robot
rgripper1 lgripper1 rgripper2 lgripper2 - gripper
room1 room2 room3 room4 room5 room6 room7 room8 - room
ball1 ball2 ball3 - obj)
(:init
(at-robby robot1 room1)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room3)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at ball1 room1)
(at ball2 room5)
(at ball3 room3)
)
(:goal
(and
(at ball1 room3)
(at ball2 room6)
(at ball3 room8)
)
)
)
