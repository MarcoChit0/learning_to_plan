

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a h)
(on b i)
(on c f)
(ontable d)
(on e a)
(on f d)
(on g b)
(on h c)
(on i e)
(clear g)
)
(:goal
(and
(on a f)
(on b d)
(on c a)
(on d g)
(on e c)
(on f h)
(on h i)
(on i b))
)
)


