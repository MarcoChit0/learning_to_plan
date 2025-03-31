(define (problem gripper-2-2-3)
(:domain gripper-strips)
(:objects robot1 robot2 - robot
rgripper1 lgripper1 rgripper2 lgripper2 - gripper
room59 room339 - room
ball343 ball372 ball441 - obj)
(:init
(at-robby robot1 room339)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room59)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at ball343 room339)
(at ball372 room339)
(at ball441 room59)
)
(:goal
(and
(at ball343 room59)
(at ball372 room59)
(at ball441 room59)
)
)
)
