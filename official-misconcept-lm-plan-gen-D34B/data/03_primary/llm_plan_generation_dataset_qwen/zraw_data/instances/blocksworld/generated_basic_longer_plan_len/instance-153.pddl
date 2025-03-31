

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(ontable a)
(on b e)
(ontable c)
(on d b)
(on e i)
(on f g)
(on g d)
(on h f)
(ontable i)
(clear a)
(clear c)
(clear h)
)
(:goal
(and
(on a e)
(on b i)
(on c f)
(on d b)
(on e d)
(on g h)
(on h a)
(on i c))
)
)


