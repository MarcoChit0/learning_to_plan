; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.5
; random seed: 654940

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child478 child3 - child
    bread88 bread34 - bread-portion
    content114 content176 - content-portion
    tray112 tray367 tray212 tray364 - tray
    table410 table290 table424 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray112 kitchen)
     (at tray367 kitchen)
     (at tray212 kitchen)
     (at tray364 kitchen)
     (at_kitchen_bread bread88)
     (at_kitchen_bread bread34)
     (at_kitchen_content content114)
     (at_kitchen_content content176)
     (no_gluten_bread bread88)
     (no_gluten_content content114)
     (allergic_gluten child478)
     (not_allergic_gluten child3)
     (waiting child478 table424)
     (waiting child3 table424)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child478)
     (served child3)
    )
  )
)
