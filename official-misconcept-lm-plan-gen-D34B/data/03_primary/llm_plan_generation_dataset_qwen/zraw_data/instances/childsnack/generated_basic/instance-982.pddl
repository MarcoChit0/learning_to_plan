; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 335317

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child53 child418 child192 - child
    bread171 bread38 bread67 - bread-portion
    content161 content329 content84 - content-portion
    tray144 - tray
    table474 table62 table161 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray144 kitchen)
     (at_kitchen_bread bread171)
     (at_kitchen_bread bread38)
     (at_kitchen_bread bread67)
     (at_kitchen_content content161)
     (at_kitchen_content content329)
     (at_kitchen_content content84)
     (not_allergic_gluten child53)
     (not_allergic_gluten child192)
     (not_allergic_gluten child418)
     (waiting child53 table161)
     (waiting child418 table62)
     (waiting child192 table161)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child53)
     (served child418)
     (served child192)
    )
  )
)
