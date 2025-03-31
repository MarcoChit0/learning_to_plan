


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d192 d309 d152 d175 )
(:init
(smaller peg1 d192)
(smaller peg1 d309)
(smaller peg1 d152)
(smaller peg1 d175)
(smaller peg2 d192)
(smaller peg2 d309)
(smaller peg2 d152)
(smaller peg2 d175)
(smaller peg3 d192)
(smaller peg3 d309)
(smaller peg3 d152)
(smaller peg3 d175)
(smaller d309 d192)
(smaller d152 d192)
(smaller d175 d192)
(smaller d152 d309)
(smaller d175 d309)
(smaller d175 d152)
(clear peg2)
(clear peg3)
(clear d192)
(on d175 peg1)
(on d152 d175)
(on d309 d152)
(on d192 d309)
)
(:goal
(and 
(on d175 peg3)
(on d152 d175)
(on d309 d152)
(on d192 d309)
)
)
)


