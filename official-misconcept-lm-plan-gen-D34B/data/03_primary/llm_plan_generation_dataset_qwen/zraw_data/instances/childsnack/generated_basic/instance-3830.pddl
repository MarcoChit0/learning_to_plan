; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 199663

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child308 - child
    bread364 - bread-portion
    content150 - content-portion
    tray353 tray367 tray58 tray101 - tray
    table116 table335 table173 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray353 kitchen)
     (at tray367 kitchen)
     (at tray58 kitchen)
     (at tray101 kitchen)
     (at_kitchen_bread bread364)
     (at_kitchen_content content150)
     (not_allergic_gluten child308)
     (waiting child308 table173)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child308)
    )
  )
)
