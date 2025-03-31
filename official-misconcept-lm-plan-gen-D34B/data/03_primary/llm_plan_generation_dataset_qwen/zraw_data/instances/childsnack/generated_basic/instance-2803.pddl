; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 359206

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child469 child130 - child
    bread100 bread39 - bread-portion
    content428 content259 - content-portion
    tray102 tray329 - tray
    table22 table393 table118 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray102 kitchen)
     (at tray329 kitchen)
     (at_kitchen_bread bread100)
     (at_kitchen_bread bread39)
     (at_kitchen_content content428)
     (at_kitchen_content content259)
     (not_allergic_gluten child469)
     (not_allergic_gluten child130)
     (waiting child469 table118)
     (waiting child130 table118)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child469)
     (served child130)
    )
  )
)
