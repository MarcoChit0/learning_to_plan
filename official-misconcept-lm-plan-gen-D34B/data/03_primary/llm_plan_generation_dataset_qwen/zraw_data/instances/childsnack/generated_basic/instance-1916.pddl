; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 745755

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child358 - child
    bread234 - bread-portion
    content440 - content-portion
    tray109 tray157 tray50 - tray
    table252 table238 table3 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray109 kitchen)
     (at tray157 kitchen)
     (at tray50 kitchen)
     (at_kitchen_bread bread234)
     (at_kitchen_content content440)
     (not_allergic_gluten child358)
     (waiting child358 table238)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child358)
    )
  )
)
