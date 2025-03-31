; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 851696

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child288 child335 - child
    bread321 bread205 - bread-portion
    content166 content137 - content-portion
    tray220 - tray
    table463 table2 table0 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray220 kitchen)
     (at_kitchen_bread bread321)
     (at_kitchen_bread bread205)
     (at_kitchen_content content166)
     (at_kitchen_content content137)
     (no_gluten_bread bread321)
     (no_gluten_content content137)
     (allergic_gluten child288)
     (not_allergic_gluten child335)
     (waiting child288 table2)
     (waiting child335 table463)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child288)
     (served child335)
    )
  )
)
