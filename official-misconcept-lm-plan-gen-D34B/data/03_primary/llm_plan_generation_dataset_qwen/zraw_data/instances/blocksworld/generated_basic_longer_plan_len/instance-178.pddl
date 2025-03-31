

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a e)
(on b h)
(on c b)
(on d i)
(on e f)
(on f g)
(on g c)
(ontable h)
(on i a)
(clear d)
)
(:goal
(and
(on a f)
(on b g)
(on c h)
(on d a)
(on e b)
(on f c)
(on h i)
(on i e))
)
)


