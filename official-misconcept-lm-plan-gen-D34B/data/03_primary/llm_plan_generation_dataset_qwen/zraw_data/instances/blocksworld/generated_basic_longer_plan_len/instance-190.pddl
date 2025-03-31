

(define (problem BW-rand-10)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i j )
(:init
(handempty)
(on a i)
(ontable b)
(on c f)
(on d a)
(on e b)
(on f j)
(on g c)
(on h d)
(on i e)
(on j h)
(clear g)
)
(:goal
(and
(on a i)
(on b c)
(on c h)
(on d g)
(on e f)
(on f b)
(on i d)
(on j e))
)
)


