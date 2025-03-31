(define (problem gripper-4-2-2)
(:domain gripper-strips)
(:objects robot1 robot2 robot3 robot4 - robot
rgripper1 lgripper1 rgripper2 lgripper2 rgripper3 lgripper3 rgripper4 lgripper4 - gripper
room358 room143 - room
ball307 ball331 - obj)
(:init
(at-robby robot1 room358)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room143)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at-robby robot3 room143)
(free robot3 rgripper3)
(free robot3 lgripper3)
(at-robby robot4 room358)
(free robot4 rgripper4)
(free robot4 lgripper4)
(at ball307 room143)
(at ball331 room358)
)
(:goal
(and
(at ball307 room358)
(at ball331 room143)
)
)
)
