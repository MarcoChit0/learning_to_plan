

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a b)
(ontable b)
(on c e)
(ontable d)
(on e i)
(on f c)
(on g a)
(on h d)
(on i g)
(clear f)
(clear h)
)
(:goal
(and
(on a e)
(on b h)
(on c b)
(on d i)
(on e c)
(on f d)
(on g a)
(on h f))
)
)


