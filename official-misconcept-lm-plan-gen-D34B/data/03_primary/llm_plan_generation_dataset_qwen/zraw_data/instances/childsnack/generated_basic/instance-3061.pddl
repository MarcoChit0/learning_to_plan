; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 32598

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child55 child495 - child
    bread169 bread360 - bread-portion
    content123 content206 - content-portion
    tray431 tray207 tray422 - tray
    table259 table492 table110 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray431 kitchen)
     (at tray207 kitchen)
     (at tray422 kitchen)
     (at_kitchen_bread bread169)
     (at_kitchen_bread bread360)
     (at_kitchen_content content123)
     (at_kitchen_content content206)
     (no_gluten_bread bread169)
     (no_gluten_content content123)
     (allergic_gluten child495)
     (not_allergic_gluten child55)
     (waiting child55 table110)
     (waiting child495 table259)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child55)
     (served child495)
    )
  )
)
