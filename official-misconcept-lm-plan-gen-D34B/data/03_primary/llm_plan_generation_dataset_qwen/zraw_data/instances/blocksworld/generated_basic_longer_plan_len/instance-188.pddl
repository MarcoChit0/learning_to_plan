

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a f)
(on b h)
(ontable c)
(on d g)
(on e d)
(on f i)
(on g a)
(on h c)
(on i b)
(clear e)
)
(:goal
(and
(on a g)
(on b a)
(on d i)
(on e d)
(on f c)
(on g h)
(on h e)
(on i f))
)
)


