(define (problem gripper-2-3-2)
(:domain gripper-strips)
(:objects robot1 robot2 - robot
rgripper1 lgripper1 rgripper2 lgripper2 - gripper
room424 room280 room3 - room
ball123 ball113 - obj)
(:init
(at-robby robot1 room280)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room280)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at ball123 room280)
(at ball113 room3)
)
(:goal
(and
(at ball123 room3)
(at ball113 room280)
)
)
)
