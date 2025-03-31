(define (problem gripper-4-2-2)
(:domain gripper-strips)
(:objects robot1 robot2 robot3 robot4 - robot
rgripper1 lgripper1 rgripper2 lgripper2 rgripper3 lgripper3 rgripper4 lgripper4 - gripper
room253 room432 - room
ball159 ball74 - obj)
(:init
(at-robby robot1 room432)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room432)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at-robby robot3 room253)
(free robot3 rgripper3)
(free robot3 lgripper3)
(at-robby robot4 room432)
(free robot4 rgripper4)
(free robot4 lgripper4)
(at ball159 room432)
(at ball74 room253)
)
(:goal
(and
(at ball159 room432)
(at ball74 room432)
)
)
)
