; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 341740

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child181 - child
    bread60 - bread-portion
    content9 - content-portion
    tray70 - tray
    table489 table482 table196 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray70 kitchen)
     (at_kitchen_bread bread60)
     (at_kitchen_content content9)
     (not_allergic_gluten child181)
     (waiting child181 table482)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child181)
    )
  )
)
