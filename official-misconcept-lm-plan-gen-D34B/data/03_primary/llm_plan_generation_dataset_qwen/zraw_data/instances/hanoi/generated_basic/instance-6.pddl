


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d392 )
(:init
(smaller peg1 d392)
(smaller peg2 d392)
(smaller peg3 d392)
(clear peg2)
(clear peg3)
(clear d392)
(on d392 peg1)
)
(:goal
(and 
(on d392 peg3)
)
)
)


