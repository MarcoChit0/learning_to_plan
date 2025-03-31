; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 367779

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child223 - child
    bread70 - bread-portion
    content482 - content-portion
    tray218 - tray
    table90 table114 table181 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray218 kitchen)
     (at_kitchen_bread bread70)
     (at_kitchen_content content482)
     (not_allergic_gluten child223)
     (waiting child223 table181)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child223)
    )
  )
)
