

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a g)
(on b d)
(on c i)
(on d f)
(on e a)
(on f h)
(on g c)
(ontable h)
(on i b)
(clear e)
)
(:goal
(and
(on a e)
(on b i)
(on d b)
(on e h)
(on f g)
(on g a)
(on i f))
)
)


