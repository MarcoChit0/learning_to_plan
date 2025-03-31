; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 57670

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child327 - child
    bread254 - bread-portion
    content112 - content-portion
    tray442 tray290 tray467 - tray
    table67 table80 table400 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray442 kitchen)
     (at tray290 kitchen)
     (at tray467 kitchen)
     (at_kitchen_bread bread254)
     (at_kitchen_content content112)
     (not_allergic_gluten child327)
     (waiting child327 table80)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child327)
    )
  )
)
