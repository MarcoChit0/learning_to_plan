; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 367780

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child373 - child
    bread308 - bread-portion
    content5 - content-portion
    tray298 tray149 tray295 - tray
    table6 table482 table204 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray298 kitchen)
     (at tray149 kitchen)
     (at tray295 kitchen)
     (at_kitchen_bread bread308)
     (at_kitchen_content content5)
     (not_allergic_gluten child373)
     (waiting child373 table6)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child373)
    )
  )
)
