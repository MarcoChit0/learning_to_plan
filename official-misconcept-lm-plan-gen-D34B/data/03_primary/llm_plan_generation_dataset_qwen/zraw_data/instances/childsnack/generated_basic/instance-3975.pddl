; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 179222

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child242 - child
    bread95 - bread-portion
    content284 - content-portion
    tray445 tray75 tray190 - tray
    table228 table91 table86 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray445 kitchen)
     (at tray75 kitchen)
     (at tray190 kitchen)
     (at_kitchen_bread bread95)
     (at_kitchen_content content284)
     (not_allergic_gluten child242)
     (waiting child242 table91)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child242)
    )
  )
)
