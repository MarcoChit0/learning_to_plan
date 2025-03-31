; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 485636

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child373 - child
    bread263 - bread-portion
    content44 - content-portion
    tray364 tray106 - tray
    table67 table226 table168 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray364 kitchen)
     (at tray106 kitchen)
     (at_kitchen_bread bread263)
     (at_kitchen_content content44)
     (not_allergic_gluten child373)
     (waiting child373 table226)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child373)
    )
  )
)
