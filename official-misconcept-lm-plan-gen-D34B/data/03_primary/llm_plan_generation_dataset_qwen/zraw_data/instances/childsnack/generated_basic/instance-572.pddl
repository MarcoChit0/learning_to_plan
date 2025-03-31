; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 610535

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child316 - child
    bread487 - bread-portion
    content177 - content-portion
    tray255 - tray
    table275 table5 table234 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray255 kitchen)
     (at_kitchen_bread bread487)
     (at_kitchen_content content177)
     (not_allergic_gluten child316)
     (waiting child316 table275)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child316)
    )
  )
)
