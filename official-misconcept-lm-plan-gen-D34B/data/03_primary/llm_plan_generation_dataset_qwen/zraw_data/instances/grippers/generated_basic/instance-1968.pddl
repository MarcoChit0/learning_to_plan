(define (problem gripper-4-2-2)
(:domain gripper-strips)
(:objects robot1 robot2 robot3 robot4 - robot
rgripper1 lgripper1 rgripper2 lgripper2 rgripper3 lgripper3 rgripper4 lgripper4 - gripper
room320 room231 - room
ball373 ball237 - obj)
(:init
(at-robby robot1 room231)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room231)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at-robby robot3 room231)
(free robot3 rgripper3)
(free robot3 lgripper3)
(at-robby robot4 room320)
(free robot4 rgripper4)
(free robot4 lgripper4)
(at ball373 room231)
(at ball237 room231)
)
(:goal
(and
(at ball373 room320)
(at ball237 room231)
)
)
)
