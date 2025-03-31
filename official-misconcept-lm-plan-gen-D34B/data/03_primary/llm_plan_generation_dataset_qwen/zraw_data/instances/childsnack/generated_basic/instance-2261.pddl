; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 832217

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child85 child109 - child
    bread126 bread10 - bread-portion
    content165 content144 - content-portion
    tray330 - tray
    table101 table2 table302 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray330 kitchen)
     (at_kitchen_bread bread126)
     (at_kitchen_bread bread10)
     (at_kitchen_content content165)
     (at_kitchen_content content144)
     (no_gluten_bread bread126)
     (no_gluten_content content165)
     (allergic_gluten child85)
     (not_allergic_gluten child109)
     (waiting child85 table2)
     (waiting child109 table2)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child85)
     (served child109)
    )
  )
)
