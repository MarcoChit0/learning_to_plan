

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a g)
(on b i)
(ontable c)
(on d b)
(on e h)
(on f c)
(on g f)
(on h a)
(on i e)
(clear d)
)
(:goal
(and
(on a c)
(on c e)
(on d g)
(on e d)
(on f i)
(on i a))
)
)


