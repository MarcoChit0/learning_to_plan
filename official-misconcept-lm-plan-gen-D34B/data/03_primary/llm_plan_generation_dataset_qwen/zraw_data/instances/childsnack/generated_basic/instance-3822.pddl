; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 519245

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child246 - child
    bread281 - bread-portion
    content153 - content-portion
    tray339 tray26 tray199 tray70 - tray
    table55 table173 table5 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray339 kitchen)
     (at tray26 kitchen)
     (at tray199 kitchen)
     (at tray70 kitchen)
     (at_kitchen_bread bread281)
     (at_kitchen_content content153)
     (not_allergic_gluten child246)
     (waiting child246 table5)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child246)
    )
  )
)
