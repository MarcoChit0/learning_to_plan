; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 726080

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child289 - child
    bread362 - bread-portion
    content483 - content-portion
    tray311 tray494 tray297 - tray
    table423 table303 table0 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray311 kitchen)
     (at tray494 kitchen)
     (at tray297 kitchen)
     (at_kitchen_bread bread362)
     (at_kitchen_content content483)
     (not_allergic_gluten child289)
     (waiting child289 table303)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child289)
    )
  )
)
