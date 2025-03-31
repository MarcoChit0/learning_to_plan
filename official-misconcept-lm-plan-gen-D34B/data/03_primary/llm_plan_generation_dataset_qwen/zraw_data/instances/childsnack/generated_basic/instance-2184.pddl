; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 424160

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child192 - child
    bread284 - bread-portion
    content63 - content-portion
    tray198 - tray
    table60 table384 table439 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray198 kitchen)
     (at_kitchen_bread bread284)
     (at_kitchen_content content63)
     (not_allergic_gluten child192)
     (waiting child192 table60)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child192)
    )
  )
)
