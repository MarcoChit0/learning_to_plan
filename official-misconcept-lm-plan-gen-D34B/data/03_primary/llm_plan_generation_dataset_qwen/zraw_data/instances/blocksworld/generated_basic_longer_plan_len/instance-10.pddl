

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a c)
(ontable b)
(on c e)
(ontable d)
(on e f)
(on f d)
(clear a)
(clear b)
)
(:goal
(and
(on a e)
(on b f)
(on c d)
(on d a)
(on f c))
)
)


