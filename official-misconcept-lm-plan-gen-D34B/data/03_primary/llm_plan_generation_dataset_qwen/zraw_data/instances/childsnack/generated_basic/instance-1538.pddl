; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 797875

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child48 child487 - child
    bread387 bread244 - bread-portion
    content234 content17 - content-portion
    tray80 tray378 tray339 tray406 - tray
    table9 table481 table36 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray80 kitchen)
     (at tray378 kitchen)
     (at tray339 kitchen)
     (at tray406 kitchen)
     (at_kitchen_bread bread387)
     (at_kitchen_bread bread244)
     (at_kitchen_content content234)
     (at_kitchen_content content17)
     (not_allergic_gluten child48)
     (not_allergic_gluten child487)
     (waiting child48 table36)
     (waiting child487 table36)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child48)
     (served child487)
    )
  )
)
