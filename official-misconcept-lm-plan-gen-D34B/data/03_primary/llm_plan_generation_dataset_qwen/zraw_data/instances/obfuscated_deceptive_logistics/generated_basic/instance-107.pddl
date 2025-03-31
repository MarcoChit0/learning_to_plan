


(define (problem obfuscated_deceptive_logistics-c2-s1-p5-a2)
(:domain obfuscated_deceptive_logistics-strips)
(:objects a0 a1 
          c0 c1 
          t0 t1 
          l0-0 l1-0 
          p0 p1 p2 p3 p4 
)
(:init
    (cats a0)
    (cats a1)
    (stupendous c0)
    (stupendous c1)
    (sneeze t0)
    (sneeze t1)
    (texture l0-0)
    (collect  l0-0 c0)
    (texture l1-0)
    (collect  l1-0 c1)
    (spring l0-0)
    (spring l1-0)
    (hand p0)
    (hand p1)
    (hand p2)
    (hand p3)
    (hand p4)
    (next t0 l0-0)
    (next t1 l1-0)
    (next p0 l1-0)
    (next p1 l1-0)
    (next p2 l0-0)
    (next p3 l1-0)
    (next p4 l1-0)
    (next a0 l0-0)
    (next a1 l1-0)
)
(:goal
    (and
        (next p0 l1-0)
        (next p1 l1-0)
        (next p2 l1-0)
        (next p3 l0-0)
        (next p4 l0-0)
    )
)
)


