(define (problem gripper-1-3-1)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room344 room237 room118 - room
ball393 - obj)
(:init
(at-robby robot1 room118)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at ball393 room118)
)
(:goal
(and
(at ball393 room344)
)
)
)
