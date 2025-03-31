; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 198183

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child147 child351 - child
    bread12 bread175 - bread-portion
    content353 content161 - content-portion
    tray206 tray234 tray408 tray261 - tray
    table173 table373 table222 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray206 kitchen)
     (at tray234 kitchen)
     (at tray408 kitchen)
     (at tray261 kitchen)
     (at_kitchen_bread bread12)
     (at_kitchen_bread bread175)
     (at_kitchen_content content353)
     (at_kitchen_content content161)
     (no_gluten_bread bread175)
     (no_gluten_content content161)
     (allergic_gluten child351)
     (not_allergic_gluten child147)
     (waiting child147 table373)
     (waiting child351 table222)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child147)
     (served child351)
    )
  )
)
