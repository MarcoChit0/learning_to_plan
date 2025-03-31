; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 134028

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child97 - child
    bread302 - bread-portion
    content106 - content-portion
    tray72 tray393 tray248 tray405 - tray
    table400 table232 table129 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray72 kitchen)
     (at tray393 kitchen)
     (at tray248 kitchen)
     (at tray405 kitchen)
     (at_kitchen_bread bread302)
     (at_kitchen_content content106)
     (not_allergic_gluten child97)
     (waiting child97 table232)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child97)
    )
  )
)
