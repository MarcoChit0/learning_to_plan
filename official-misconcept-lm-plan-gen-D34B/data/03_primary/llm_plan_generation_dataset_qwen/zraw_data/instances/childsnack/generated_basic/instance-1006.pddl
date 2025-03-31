; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 321195

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child196 child315 child104 - child
    bread148 bread358 bread234 - bread-portion
    content469 content217 content412 - content-portion
    tray138 - tray
    table447 table347 table161 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray138 kitchen)
     (at_kitchen_bread bread148)
     (at_kitchen_bread bread358)
     (at_kitchen_bread bread234)
     (at_kitchen_content content469)
     (at_kitchen_content content217)
     (at_kitchen_content content412)
     (not_allergic_gluten child315)
     (not_allergic_gluten child196)
     (not_allergic_gluten child104)
     (waiting child196 table347)
     (waiting child315 table347)
     (waiting child104 table447)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child196)
     (served child315)
     (served child104)
    )
  )
)
