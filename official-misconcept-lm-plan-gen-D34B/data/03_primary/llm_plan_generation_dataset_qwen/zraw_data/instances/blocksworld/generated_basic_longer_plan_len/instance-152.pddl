

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a f)
(on b h)
(ontable c)
(on d e)
(on e a)
(on f i)
(on g b)
(ontable h)
(on i g)
(clear c)
(clear d)
)
(:goal
(and
(on a c)
(on c g)
(on d f)
(on e b)
(on f h)
(on h i)
(on i a))
)
)


