; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 779334

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child243 child202 - child
    bread66 bread131 - bread-portion
    content474 content198 - content-portion
    tray443 tray234 - tray
    table321 table276 table59 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray443 kitchen)
     (at tray234 kitchen)
     (at_kitchen_bread bread66)
     (at_kitchen_bread bread131)
     (at_kitchen_content content474)
     (at_kitchen_content content198)
     (no_gluten_bread bread131)
     (no_gluten_content content198)
     (allergic_gluten child243)
     (not_allergic_gluten child202)
     (waiting child243 table321)
     (waiting child202 table59)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child243)
     (served child202)
    )
  )
)
