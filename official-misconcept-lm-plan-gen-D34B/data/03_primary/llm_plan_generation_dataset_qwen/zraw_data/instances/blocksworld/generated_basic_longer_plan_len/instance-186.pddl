

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a e)
(on b h)
(on c d)
(on d a)
(on e i)
(on f c)
(ontable g)
(on h g)
(on i b)
(clear f)
)
(:goal
(and
(on a e)
(on b h)
(on c a)
(on d b)
(on f g)
(on g i)
(on h f)
(on i c))
)
)


