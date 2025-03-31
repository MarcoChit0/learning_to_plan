

(define (problem BW-rand-10)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i j )
(:init
(handempty)
(on a i)
(on b j)
(ontable c)
(ontable d)
(on e h)
(on f g)
(on g b)
(ontable h)
(on i f)
(on j d)
(clear a)
(clear c)
(clear e)
)
(:goal
(and
(on a d)
(on b a)
(on c f)
(on e h)
(on f e)
(on g b)
(on h j)
(on i g))
)
)


