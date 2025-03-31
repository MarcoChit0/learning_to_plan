


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d424 d113 )
(:init
(smaller peg1 d424)
(smaller peg1 d113)
(smaller peg2 d424)
(smaller peg2 d113)
(smaller peg3 d424)
(smaller peg3 d113)
(smaller d113 d424)
(clear peg2)
(clear peg3)
(clear d424)
(on d113 peg1)
(on d424 d113)
)
(:goal
(and 
(on d113 peg3)
(on d424 d113)
)
)
)


