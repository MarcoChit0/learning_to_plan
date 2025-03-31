; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 24306

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child53 - child
    bread44 - bread-portion
    content95 - content-portion
    tray318 - tray
    table109 table276 table164 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray318 kitchen)
     (at_kitchen_bread bread44)
     (at_kitchen_content content95)
     (not_allergic_gluten child53)
     (waiting child53 table164)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child53)
    )
  )
)
