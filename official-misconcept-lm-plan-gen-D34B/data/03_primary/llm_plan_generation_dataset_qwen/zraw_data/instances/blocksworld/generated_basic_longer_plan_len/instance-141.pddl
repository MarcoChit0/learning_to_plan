

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a c)
(on b h)
(on c d)
(on d i)
(on e f)
(ontable f)
(on g e)
(on h g)
(on i b)
(clear a)
)
(:goal
(and
(on a b)
(on b h)
(on d c)
(on e f)
(on f d)
(on h i)
(on i e))
)
)


