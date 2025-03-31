; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 92835

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child421 - child
    bread377 - bread-portion
    content367 - content-portion
    tray369 - tray
    table129 table475 table105 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray369 kitchen)
     (at_kitchen_bread bread377)
     (at_kitchen_content content367)
     (not_allergic_gluten child421)
     (waiting child421 table105)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child421)
    )
  )
)
