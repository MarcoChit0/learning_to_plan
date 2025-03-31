; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 657382

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child459 child227 child479 - child
    bread169 bread458 bread147 - bread-portion
    content111 content256 content126 - content-portion
    tray1 - tray
    table321 table211 table469 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray1 kitchen)
     (at_kitchen_bread bread169)
     (at_kitchen_bread bread458)
     (at_kitchen_bread bread147)
     (at_kitchen_content content111)
     (at_kitchen_content content256)
     (at_kitchen_content content126)
     (not_allergic_gluten child479)
     (not_allergic_gluten child459)
     (not_allergic_gluten child227)
     (waiting child459 table469)
     (waiting child227 table211)
     (waiting child479 table469)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child459)
     (served child227)
     (served child479)
    )
  )
)
