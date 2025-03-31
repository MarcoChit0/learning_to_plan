; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 149753

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child189 child40 - child
    bread169 bread70 - bread-portion
    content308 content107 - content-portion
    tray60 tray115 tray467 - tray
    table174 table321 table157 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray60 kitchen)
     (at tray115 kitchen)
     (at tray467 kitchen)
     (at_kitchen_bread bread169)
     (at_kitchen_bread bread70)
     (at_kitchen_content content308)
     (at_kitchen_content content107)
     (not_allergic_gluten child189)
     (not_allergic_gluten child40)
     (waiting child189 table157)
     (waiting child40 table157)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child189)
     (served child40)
    )
  )
)
