

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a g)
(ontable b)
(on c i)
(on d e)
(on e c)
(on f a)
(on g d)
(ontable h)
(on i h)
(clear b)
(clear f)
)
(:goal
(and
(on a e)
(on b a)
(on c d)
(on d i)
(on f g)
(on g b)
(on h f)
(on i h))
)
)


