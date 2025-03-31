

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a d)
(on b g)
(on c f)
(on d h)
(ontable e)
(on f e)
(on g i)
(on h c)
(on i a)
(clear b)
)
(:goal
(and
(on a d)
(on b h)
(on c b)
(on d e)
(on e i)
(on h a)
(on i f))
)
)


