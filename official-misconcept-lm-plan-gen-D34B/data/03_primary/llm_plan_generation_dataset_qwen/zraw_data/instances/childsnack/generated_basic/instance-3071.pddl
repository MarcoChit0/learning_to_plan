; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 561590

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child412 child305 - child
    bread14 bread22 - bread-portion
    content180 content89 - content-portion
    tray417 tray445 tray443 - tray
    table83 table429 table114 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray417 kitchen)
     (at tray445 kitchen)
     (at tray443 kitchen)
     (at_kitchen_bread bread14)
     (at_kitchen_bread bread22)
     (at_kitchen_content content180)
     (at_kitchen_content content89)
     (no_gluten_bread bread22)
     (no_gluten_content content89)
     (allergic_gluten child412)
     (not_allergic_gluten child305)
     (waiting child412 table429)
     (waiting child305 table429)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child412)
     (served child305)
    )
  )
)
