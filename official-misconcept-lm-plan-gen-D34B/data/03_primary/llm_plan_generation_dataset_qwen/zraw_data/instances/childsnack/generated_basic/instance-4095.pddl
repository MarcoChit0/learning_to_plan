; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 873535

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child229 - child
    bread232 - bread-portion
    content253 - content-portion
    tray406 tray454 tray314 - tray
    table299 table341 table171 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray406 kitchen)
     (at tray454 kitchen)
     (at tray314 kitchen)
     (at_kitchen_bread bread232)
     (at_kitchen_content content253)
     (not_allergic_gluten child229)
     (waiting child229 table299)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child229)
    )
  )
)
