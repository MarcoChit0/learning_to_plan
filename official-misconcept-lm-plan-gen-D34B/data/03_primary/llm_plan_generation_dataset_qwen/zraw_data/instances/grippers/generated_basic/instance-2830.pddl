(define (problem gripper-2-2-3)
(:domain gripper-strips)
(:objects robot1 robot2 - robot
rgripper1 lgripper1 rgripper2 lgripper2 - gripper
room383 room94 - room
ball373 ball395 ball426 - obj)
(:init
(at-robby robot1 room94)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room94)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at ball373 room94)
(at ball395 room94)
(at ball426 room383)
)
(:goal
(and
(at ball373 room383)
(at ball395 room383)
(at ball426 room94)
)
)
)
