(define (problem gripper-4-2-2)
(:domain gripper-strips)
(:objects robot1 robot2 robot3 robot4 - robot
rgripper1 lgripper1 rgripper2 lgripper2 rgripper3 lgripper3 rgripper4 lgripper4 - gripper
room251 room103 - room
ball404 ball415 - obj)
(:init
(at-robby robot1 room251)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room103)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at-robby robot3 room251)
(free robot3 rgripper3)
(free robot3 lgripper3)
(at-robby robot4 room251)
(free robot4 rgripper4)
(free robot4 lgripper4)
(at ball404 room103)
(at ball415 room251)
)
(:goal
(and
(at ball404 room251)
(at ball415 room251)
)
)
)
