; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 473806

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child112 - child
    bread303 - bread-portion
    content127 - content-portion
    tray112 tray255 tray334 - tray
    table422 table131 table77 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray112 kitchen)
     (at tray255 kitchen)
     (at tray334 kitchen)
     (at_kitchen_bread bread303)
     (at_kitchen_content content127)
     (not_allergic_gluten child112)
     (waiting child112 table422)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child112)
    )
  )
)
