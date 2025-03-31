; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 180883

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child308 child104 child178 - child
    bread351 bread397 bread152 - bread-portion
    content442 content391 content145 - content-portion
    tray450 - tray
    table420 table111 table229 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray450 kitchen)
     (at_kitchen_bread bread351)
     (at_kitchen_bread bread397)
     (at_kitchen_bread bread152)
     (at_kitchen_content content442)
     (at_kitchen_content content391)
     (at_kitchen_content content145)
     (not_allergic_gluten child104)
     (not_allergic_gluten child308)
     (not_allergic_gluten child178)
     (waiting child308 table420)
     (waiting child104 table229)
     (waiting child178 table229)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child308)
     (served child104)
     (served child178)
    )
  )
)
