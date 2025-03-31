; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 667678

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child73 - child
    bread154 - bread-portion
    content152 - content-portion
    tray132 - tray
    table483 table109 table78 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray132 kitchen)
     (at_kitchen_bread bread154)
     (at_kitchen_content content152)
     (not_allergic_gluten child73)
     (waiting child73 table78)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child73)
    )
  )
)
