

(define (problem BW-rand-10)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i j )
(:init
(handempty)
(on a d)
(on b g)
(on c b)
(ontable d)
(ontable e)
(on f e)
(on g i)
(on h f)
(on i h)
(on j c)
(clear a)
(clear j)
)
(:goal
(and
(on a g)
(on b i)
(on c h)
(on e j)
(on g b)
(on h d)
(on i f)
(on j a))
)
)


