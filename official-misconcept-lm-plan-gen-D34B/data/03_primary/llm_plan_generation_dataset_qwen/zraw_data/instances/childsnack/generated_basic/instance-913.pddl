; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 985869

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 child99 child478 - child
    bread136 bread308 bread294 - bread-portion
    content154 content188 content234 - content-portion
    tray222 - tray
    table173 table155 table381 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray222 kitchen)
     (at_kitchen_bread bread136)
     (at_kitchen_bread bread308)
     (at_kitchen_bread bread294)
     (at_kitchen_content content154)
     (at_kitchen_content content188)
     (at_kitchen_content content234)
     (not_allergic_gluten child99)
     (not_allergic_gluten child431)
     (not_allergic_gluten child478)
     (waiting child431 table173)
     (waiting child99 table155)
     (waiting child478 table155)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child431)
     (served child99)
     (served child478)
    )
  )
)
