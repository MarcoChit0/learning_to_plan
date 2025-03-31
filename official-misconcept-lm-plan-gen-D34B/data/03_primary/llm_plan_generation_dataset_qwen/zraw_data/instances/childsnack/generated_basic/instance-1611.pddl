; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 861589

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child217 - child
    bread76 - bread-portion
    content254 - content-portion
    tray380 tray33 tray51 tray0 - tray
    table135 table173 table158 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray380 kitchen)
     (at tray33 kitchen)
     (at tray51 kitchen)
     (at tray0 kitchen)
     (at_kitchen_bread bread76)
     (at_kitchen_content content254)
     (not_allergic_gluten child217)
     (waiting child217 table135)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child217)
    )
  )
)
