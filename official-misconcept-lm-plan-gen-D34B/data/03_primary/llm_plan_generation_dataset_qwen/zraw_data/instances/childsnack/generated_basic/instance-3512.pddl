; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 29648

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child481 child5 - child
    bread103 bread233 - bread-portion
    content358 content42 - content-portion
    tray173 tray66 - tray
    table495 table0 table228 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray173 kitchen)
     (at tray66 kitchen)
     (at_kitchen_bread bread103)
     (at_kitchen_bread bread233)
     (at_kitchen_content content358)
     (at_kitchen_content content42)
     (not_allergic_gluten child5)
     (not_allergic_gluten child481)
     (waiting child481 table495)
     (waiting child5 table228)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child481)
     (served child5)
    )
  )
)
