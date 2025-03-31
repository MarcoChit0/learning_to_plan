


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d379 d428 d496 d452 )
(:init
(smaller peg1 d379)
(smaller peg1 d428)
(smaller peg1 d496)
(smaller peg1 d452)
(smaller peg2 d379)
(smaller peg2 d428)
(smaller peg2 d496)
(smaller peg2 d452)
(smaller peg3 d379)
(smaller peg3 d428)
(smaller peg3 d496)
(smaller peg3 d452)
(smaller d428 d379)
(smaller d496 d379)
(smaller d452 d379)
(smaller d496 d428)
(smaller d452 d428)
(smaller d452 d496)
(clear peg2)
(clear peg3)
(clear d379)
(on d452 peg1)
(on d496 d452)
(on d428 d496)
(on d379 d428)
)
(:goal
(and 
(on d452 peg3)
(on d496 d452)
(on d428 d496)
(on d379 d428)
)
)
)


