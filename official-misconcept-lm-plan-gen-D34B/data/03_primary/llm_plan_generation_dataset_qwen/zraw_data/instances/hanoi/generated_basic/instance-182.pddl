


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d11 d305 d197 d401 )
(:init
(smaller peg1 d11)
(smaller peg1 d305)
(smaller peg1 d197)
(smaller peg1 d401)
(smaller peg2 d11)
(smaller peg2 d305)
(smaller peg2 d197)
(smaller peg2 d401)
(smaller peg3 d11)
(smaller peg3 d305)
(smaller peg3 d197)
(smaller peg3 d401)
(smaller d305 d11)
(smaller d197 d11)
(smaller d401 d11)
(smaller d197 d305)
(smaller d401 d305)
(smaller d401 d197)
(clear peg2)
(clear peg3)
(clear d11)
(on d401 peg1)
(on d197 d401)
(on d305 d197)
(on d11 d305)
)
(:goal
(and 
(on d401 peg3)
(on d197 d401)
(on d305 d197)
(on d11 d305)
)
)
)


