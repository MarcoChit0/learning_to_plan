

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a c)
(on b d)
(ontable c)
(on d g)
(on e a)
(on f b)
(on g e)
(clear f)
)
(:goal
(and
(on b d)
(on c g)
(on d f)
(on f c)
(on g e))
)
)


