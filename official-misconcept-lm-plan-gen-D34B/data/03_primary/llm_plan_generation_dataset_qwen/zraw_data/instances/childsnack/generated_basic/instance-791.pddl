; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 63570

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child17 - child
    bread256 - bread-portion
    content0 - content-portion
    tray67 tray1 - tray
    table350 table187 table96 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray67 kitchen)
     (at tray1 kitchen)
     (at_kitchen_bread bread256)
     (at_kitchen_content content0)
     (not_allergic_gluten child17)
     (waiting child17 table187)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child17)
    )
  )
)
