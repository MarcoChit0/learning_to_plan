; child-snack task with 2 children and 0.72 gluten factor 
; constant factor of 1.7
; random seed: 476334

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child139 child435 - child
    bread281 bread344 - bread-portion
    content88 content343 - content-portion
    tray474 tray18 tray328 - tray
    table436 table45 table327 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray474 kitchen)
     (at tray18 kitchen)
     (at tray328 kitchen)
     (at_kitchen_bread bread281)
     (at_kitchen_bread bread344)
     (at_kitchen_content content88)
     (at_kitchen_content content343)
     (no_gluten_bread bread281)
     (no_gluten_content content343)
     (allergic_gluten child139)
     (not_allergic_gluten child435)
     (waiting child139 table327)
     (waiting child435 table45)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child139)
     (served child435)
    )
  )
)
