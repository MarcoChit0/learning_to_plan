

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a h)
(on b g)
(on c i)
(ontable d)
(ontable e)
(on f e)
(on g a)
(on h d)
(on i f)
(clear b)
(clear c)
)
(:goal
(and
(on a e)
(on b d)
(on c i)
(on d h)
(on e c)
(on f a)
(on g b)
(on i g))
)
)


