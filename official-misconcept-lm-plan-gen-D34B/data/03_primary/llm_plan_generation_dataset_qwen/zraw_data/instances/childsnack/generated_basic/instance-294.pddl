; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 156562

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child8 - child
    bread258 - bread-portion
    content480 - content-portion
    tray142 - tray
    table161 table361 table5 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray142 kitchen)
     (at_kitchen_bread bread258)
     (at_kitchen_content content480)
     (not_allergic_gluten child8)
     (waiting child8 table361)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child8)
    )
  )
)
