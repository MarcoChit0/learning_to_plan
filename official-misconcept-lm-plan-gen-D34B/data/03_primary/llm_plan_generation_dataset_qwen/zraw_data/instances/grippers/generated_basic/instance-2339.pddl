(define (problem gripper-2-2-1)
(:domain gripper-strips)
(:objects robot1 robot2 - robot
rgripper1 lgripper1 rgripper2 lgripper2 - gripper
room172 room38 - room
ball336 - obj)
(:init
(at-robby robot1 room38)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room38)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at ball336 room38)
)
(:goal
(and
(at ball336 room172)
)
)
)
