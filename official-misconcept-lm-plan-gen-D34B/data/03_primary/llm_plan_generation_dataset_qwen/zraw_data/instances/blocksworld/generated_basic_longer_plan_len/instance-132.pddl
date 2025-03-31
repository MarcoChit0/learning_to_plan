

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(ontable a)
(on b i)
(on c a)
(on d c)
(on e d)
(on f b)
(on g h)
(on h e)
(ontable i)
(clear f)
(clear g)
)
(:goal
(and
(on a e)
(on b g)
(on d i)
(on e c)
(on f d)
(on g h)
(on h a)
(on i b))
)
)


