

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a c)
(on b e)
(on c f)
(ontable d)
(on e i)
(on f g)
(ontable g)
(on h d)
(on i a)
(clear b)
(clear h)
)
(:goal
(and
(on a i)
(on b e)
(on d b)
(on e c)
(on f h)
(on g a)
(on h g)
(on i d))
)
)


