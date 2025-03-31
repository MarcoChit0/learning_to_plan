

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a f)
(on b i)
(on c b)
(on d h)
(on e c)
(on f d)
(on g e)
(ontable h)
(ontable i)
(clear a)
(clear g)
)
(:goal
(and
(on a h)
(on c g)
(on d e)
(on e i)
(on f c)
(on g d)
(on h f)
(on i b))
)
)


