; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.3
; random seed: 818180

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child421 child318 - child
    bread293 bread379 - bread-portion
    content210 content452 - content-portion
    tray18 tray436 - tray
    table340 table280 table26 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray18 kitchen)
     (at tray436 kitchen)
     (at_kitchen_bread bread293)
     (at_kitchen_bread bread379)
     (at_kitchen_content content210)
     (at_kitchen_content content452)
     (not_allergic_gluten child421)
     (not_allergic_gluten child318)
     (waiting child421 table26)
     (waiting child318 table280)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child421)
     (served child318)
    )
  )
)
