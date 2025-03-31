

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a g)
(on b c)
(on c f)
(on d b)
(ontable e)
(on f e)
(on g d)
(on h a)
(clear h)
)
(:goal
(and
(on a f)
(on b g)
(on d a)
(on e c)
(on f h)
(on g e))
)
)


