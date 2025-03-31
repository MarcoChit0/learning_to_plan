

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a g)
(on b i)
(on c b)
(on d c)
(on e d)
(on f e)
(on g f)
(ontable h)
(on i h)
(clear a)
)
(:goal
(and
(on a h)
(on b d)
(on c e)
(on d a)
(on e b)
(on g i)
(on h f))
)
)


