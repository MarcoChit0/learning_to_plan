; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 694737

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child115 - child
    bread169 - bread-portion
    content183 - content-portion
    tray280 tray229 - tray
    table165 table102 table0 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray280 kitchen)
     (at tray229 kitchen)
     (at_kitchen_bread bread169)
     (at_kitchen_content content183)
     (not_allergic_gluten child115)
     (waiting child115 table0)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child115)
    )
  )
)
