(define (problem gripper-5-9-1)
(:domain gripper-strips)
(:objects robot1 robot2 robot3 robot4 robot5 - robot
rgripper1 lgripper1 rgripper2 lgripper2 rgripper3 lgripper3 rgripper4 lgripper4 rgripper5 lgripper5 - gripper
room1 room2 room3 room4 room5 room6 room7 room8 room9 - room
ball1 - obj)
(:init
(at-robby robot1 room2)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room6)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at-robby robot3 room5)
(free robot3 rgripper3)
(free robot3 lgripper3)
(at-robby robot4 room9)
(free robot4 rgripper4)
(free robot4 lgripper4)
(at-robby robot5 room2)
(free robot5 rgripper5)
(free robot5 lgripper5)
(at ball1 room4)
)
(:goal
(and
(at ball1 room3)
)
)
)
