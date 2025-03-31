; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 910556

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child483 - child
    bread376 - bread-portion
    content43 - content-portion
    tray289 - tray
    table390 table302 table387 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray289 kitchen)
     (at_kitchen_bread bread376)
     (at_kitchen_content content43)
     (not_allergic_gluten child483)
     (waiting child483 table302)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child483)
    )
  )
)
