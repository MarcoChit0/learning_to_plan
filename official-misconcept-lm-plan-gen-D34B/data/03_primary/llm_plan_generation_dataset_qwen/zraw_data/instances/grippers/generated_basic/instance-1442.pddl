(define (problem gripper-3-2-2)
(:domain gripper-strips)
(:objects robot1 robot2 robot3 - robot
rgripper1 lgripper1 rgripper2 lgripper2 rgripper3 lgripper3 - gripper
room430 room391 - room
ball210 ball86 - obj)
(:init
(at-robby robot1 room391)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room430)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at-robby robot3 room430)
(free robot3 rgripper3)
(free robot3 lgripper3)
(at ball210 room391)
(at ball86 room430)
)
(:goal
(and
(at ball210 room430)
(at ball86 room391)
)
)
)
