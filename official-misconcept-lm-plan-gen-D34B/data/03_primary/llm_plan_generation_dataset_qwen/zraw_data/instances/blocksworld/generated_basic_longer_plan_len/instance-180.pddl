

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a e)
(on b f)
(on c a)
(on d c)
(on e i)
(ontable f)
(on g h)
(on h b)
(on i g)
(clear d)
)
(:goal
(and
(on a f)
(on b g)
(on c b)
(on d c)
(on e i)
(on h a)
(on i h))
)
)


