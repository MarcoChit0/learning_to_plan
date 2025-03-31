; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 835316

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child38 child331 child125 - child
    bread444 bread152 bread58 - bread-portion
    content259 content108 content133 - content-portion
    tray80 tray87 tray307 - tray
    table120 table3 table135 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray80 kitchen)
     (at tray87 kitchen)
     (at tray307 kitchen)
     (at_kitchen_bread bread444)
     (at_kitchen_bread bread152)
     (at_kitchen_bread bread58)
     (at_kitchen_content content259)
     (at_kitchen_content content108)
     (at_kitchen_content content133)
     (not_allergic_gluten child331)
     (not_allergic_gluten child125)
     (not_allergic_gluten child38)
     (waiting child38 table120)
     (waiting child331 table120)
     (waiting child125 table135)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child38)
     (served child331)
     (served child125)
    )
  )
)
