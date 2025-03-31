; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 284602

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child217 child291 - child
    bread426 bread82 - bread-portion
    content114 content37 - content-portion
    tray438 tray314 tray91 - tray
    table32 table2 table345 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray438 kitchen)
     (at tray314 kitchen)
     (at tray91 kitchen)
     (at_kitchen_bread bread426)
     (at_kitchen_bread bread82)
     (at_kitchen_content content114)
     (at_kitchen_content content37)
     (not_allergic_gluten child217)
     (not_allergic_gluten child291)
     (waiting child217 table2)
     (waiting child291 table2)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child217)
     (served child291)
    )
  )
)
