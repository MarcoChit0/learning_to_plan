; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 394304

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child109 child104 - child
    bread28 bread234 - bread-portion
    content48 content138 - content-portion
    tray34 tray75 tray195 - tray
    table35 table56 table309 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray34 kitchen)
     (at tray75 kitchen)
     (at tray195 kitchen)
     (at_kitchen_bread bread28)
     (at_kitchen_bread bread234)
     (at_kitchen_content content48)
     (at_kitchen_content content138)
     (not_allergic_gluten child109)
     (not_allergic_gluten child104)
     (waiting child109 table35)
     (waiting child104 table309)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child109)
     (served child104)
    )
  )
)
