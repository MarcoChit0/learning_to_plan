; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 407276

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child465 child470 child403 - child
    bread110 bread250 bread456 - bread-portion
    content49 content100 content445 - content-portion
    tray87 - tray
    table298 table228 table192 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray87 kitchen)
     (at_kitchen_bread bread110)
     (at_kitchen_bread bread250)
     (at_kitchen_bread bread456)
     (at_kitchen_content content49)
     (at_kitchen_content content100)
     (at_kitchen_content content445)
     (not_allergic_gluten child403)
     (not_allergic_gluten child465)
     (not_allergic_gluten child470)
     (waiting child465 table298)
     (waiting child470 table228)
     (waiting child403 table192)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child465)
     (served child470)
     (served child403)
    )
  )
)
