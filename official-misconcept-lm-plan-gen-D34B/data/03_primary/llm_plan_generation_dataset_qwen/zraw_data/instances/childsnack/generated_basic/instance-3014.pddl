; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 748960

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child423 child308 - child
    bread469 bread339 - bread-portion
    content422 content283 - content-portion
    tray374 tray9 tray294 tray18 - tray
    table113 table213 table387 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray374 kitchen)
     (at tray9 kitchen)
     (at tray294 kitchen)
     (at tray18 kitchen)
     (at_kitchen_bread bread469)
     (at_kitchen_bread bread339)
     (at_kitchen_content content422)
     (at_kitchen_content content283)
     (not_allergic_gluten child308)
     (not_allergic_gluten child423)
     (waiting child423 table113)
     (waiting child308 table387)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child423)
     (served child308)
    )
  )
)
