; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 388923

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child17 - child
    bread335 - bread-portion
    content319 - content-portion
    tray479 tray250 tray405 - tray
    table130 table396 table353 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray479 kitchen)
     (at tray250 kitchen)
     (at tray405 kitchen)
     (at_kitchen_bread bread335)
     (at_kitchen_content content319)
     (not_allergic_gluten child17)
     (waiting child17 table396)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child17)
    )
  )
)
