


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d248 d146 d143 )
(:init
(smaller peg1 d248)
(smaller peg1 d146)
(smaller peg1 d143)
(smaller peg2 d248)
(smaller peg2 d146)
(smaller peg2 d143)
(smaller peg3 d248)
(smaller peg3 d146)
(smaller peg3 d143)
(smaller d146 d248)
(smaller d143 d248)
(smaller d143 d146)
(clear peg2)
(clear peg3)
(clear d248)
(on d143 peg1)
(on d146 d143)
(on d248 d146)
)
(:goal
(and 
(on d143 peg3)
(on d146 d143)
(on d248 d146)
)
)
)


