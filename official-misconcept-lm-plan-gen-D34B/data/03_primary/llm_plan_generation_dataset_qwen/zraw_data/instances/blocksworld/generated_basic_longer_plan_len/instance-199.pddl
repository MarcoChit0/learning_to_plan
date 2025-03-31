

(define (problem BW-rand-10)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i j )
(:init
(handempty)
(on a d)
(on b c)
(ontable c)
(on d j)
(on e f)
(ontable f)
(ontable g)
(on h i)
(on i e)
(on j b)
(clear a)
(clear g)
(clear h)
)
(:goal
(and
(on a e)
(on c h)
(on d j)
(on e d)
(on f b)
(on g a)
(on h i)
(on i f)
(on j c))
)
)


