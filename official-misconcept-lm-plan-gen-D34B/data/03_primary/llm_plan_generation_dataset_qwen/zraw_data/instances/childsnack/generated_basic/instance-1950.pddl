; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 236113

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child252 - child
    bread227 - bread-portion
    content147 - content-portion
    tray195 tray26 tray424 - tray
    table19 table11 table447 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray195 kitchen)
     (at tray26 kitchen)
     (at tray424 kitchen)
     (at_kitchen_bread bread227)
     (at_kitchen_content content147)
     (not_allergic_gluten child252)
     (waiting child252 table447)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child252)
    )
  )
)
