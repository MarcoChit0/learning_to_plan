

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a c)
(on b f)
(on c g)
(on d i)
(on e h)
(on f a)
(ontable g)
(on h d)
(on i b)
(clear e)
)
(:goal
(and
(on a i)
(on b f)
(on c e)
(on d h)
(on e a)
(on f c)
(on i g))
)
)


