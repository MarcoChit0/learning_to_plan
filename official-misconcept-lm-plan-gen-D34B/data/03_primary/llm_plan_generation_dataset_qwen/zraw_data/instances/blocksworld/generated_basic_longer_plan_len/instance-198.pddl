

(define (problem BW-rand-10)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i j )
(:init
(handempty)
(on a j)
(ontable b)
(ontable c)
(on d e)
(ontable e)
(on f b)
(on g f)
(on h g)
(on i a)
(on j h)
(clear c)
(clear d)
(clear i)
)
(:goal
(and
(on a j)
(on b a)
(on c b)
(on d i)
(on e d)
(on g h)
(on h f)
(on i c)
(on j g))
)
)


