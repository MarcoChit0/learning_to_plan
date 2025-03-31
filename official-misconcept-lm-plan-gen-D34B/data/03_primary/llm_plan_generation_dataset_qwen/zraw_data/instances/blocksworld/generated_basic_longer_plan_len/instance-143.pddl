

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a b)
(on b i)
(ontable c)
(ontable d)
(on e d)
(ontable f)
(on g a)
(on h g)
(on i e)
(clear c)
(clear f)
(clear h)
)
(:goal
(and
(on a g)
(on c d)
(on d b)
(on e c)
(on f i)
(on g e)
(on h a)
(on i h))
)
)


