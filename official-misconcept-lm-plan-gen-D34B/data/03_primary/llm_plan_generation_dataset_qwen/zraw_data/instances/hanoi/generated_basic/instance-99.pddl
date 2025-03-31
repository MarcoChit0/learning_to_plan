


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d212 d0 )
(:init
(smaller peg1 d212)
(smaller peg1 d0)
(smaller peg2 d212)
(smaller peg2 d0)
(smaller peg3 d212)
(smaller peg3 d0)
(smaller d0 d212)
(clear peg2)
(clear peg3)
(clear d212)
(on d0 peg1)
(on d212 d0)
)
(:goal
(and 
(on d0 peg3)
(on d212 d0)
)
)
)


