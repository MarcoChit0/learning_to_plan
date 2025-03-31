


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d169 d143 d208 )
(:init
(smaller peg1 d169)
(smaller peg1 d143)
(smaller peg1 d208)
(smaller peg2 d169)
(smaller peg2 d143)
(smaller peg2 d208)
(smaller peg3 d169)
(smaller peg3 d143)
(smaller peg3 d208)
(smaller d143 d169)
(smaller d208 d169)
(smaller d208 d143)
(clear peg2)
(clear peg3)
(clear d169)
(on d208 peg1)
(on d143 d208)
(on d169 d143)
)
(:goal
(and 
(on d208 peg3)
(on d143 d208)
(on d169 d143)
)
)
)


