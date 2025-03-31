(define (problem gripper-2-2-1)
(:domain gripper-strips)
(:objects robot1 robot2 - robot
rgripper1 lgripper1 rgripper2 lgripper2 - gripper
room197 room255 - room
ball355 - obj)
(:init
(at-robby robot1 room197)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room255)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at ball355 room255)
)
(:goal
(and
(at ball355 room197)
)
)
)
