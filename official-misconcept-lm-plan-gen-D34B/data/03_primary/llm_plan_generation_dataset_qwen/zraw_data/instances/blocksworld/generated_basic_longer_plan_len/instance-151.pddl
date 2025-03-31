

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(ontable a)
(on b c)
(on c a)
(on d h)
(on e f)
(on f b)
(on g d)
(on h i)
(on i e)
(clear g)
)
(:goal
(and
(on a e)
(on c d)
(on d b)
(on e c)
(on f a)
(on g f)
(on h i)
(on i g))
)
)


