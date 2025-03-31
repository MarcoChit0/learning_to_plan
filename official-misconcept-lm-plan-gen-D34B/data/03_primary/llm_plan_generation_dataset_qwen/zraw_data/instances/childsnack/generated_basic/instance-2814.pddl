; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 19539

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child373 child300 - child
    bread418 bread139 - bread-portion
    content194 content360 - content-portion
    tray339 tray317 - tray
    table107 table129 table109 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray339 kitchen)
     (at tray317 kitchen)
     (at_kitchen_bread bread418)
     (at_kitchen_bread bread139)
     (at_kitchen_content content194)
     (at_kitchen_content content360)
     (not_allergic_gluten child373)
     (not_allergic_gluten child300)
     (waiting child373 table129)
     (waiting child300 table107)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child373)
     (served child300)
    )
  )
)
