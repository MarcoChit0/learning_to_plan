; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 323371

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child81 - child
    bread61 - bread-portion
    content383 - content-portion
    tray369 tray223 tray314 - tray
    table468 table217 table284 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray369 kitchen)
     (at tray223 kitchen)
     (at tray314 kitchen)
     (at_kitchen_bread bread61)
     (at_kitchen_content content383)
     (not_allergic_gluten child81)
     (waiting child81 table468)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child81)
    )
  )
)
