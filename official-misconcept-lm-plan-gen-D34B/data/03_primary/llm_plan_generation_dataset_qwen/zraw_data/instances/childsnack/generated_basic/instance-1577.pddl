; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 190679

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child109 child176 - child
    bread251 bread139 - bread-portion
    content284 content129 - content-portion
    tray35 tray498 tray435 tray454 - tray
    table111 table150 table335 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray35 kitchen)
     (at tray498 kitchen)
     (at tray435 kitchen)
     (at tray454 kitchen)
     (at_kitchen_bread bread251)
     (at_kitchen_bread bread139)
     (at_kitchen_content content284)
     (at_kitchen_content content129)
     (not_allergic_gluten child176)
     (not_allergic_gluten child109)
     (waiting child109 table111)
     (waiting child176 table150)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child109)
     (served child176)
    )
  )
)
