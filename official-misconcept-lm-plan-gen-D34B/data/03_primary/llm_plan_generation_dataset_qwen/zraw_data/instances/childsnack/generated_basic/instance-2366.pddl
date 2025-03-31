; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 593915

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child138 child469 - child
    bread40 bread65 - bread-portion
    content84 content111 - content-portion
    tray59 - tray
    table300 table425 table29 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray59 kitchen)
     (at_kitchen_bread bread40)
     (at_kitchen_bread bread65)
     (at_kitchen_content content84)
     (at_kitchen_content content111)
     (not_allergic_gluten child138)
     (not_allergic_gluten child469)
     (waiting child138 table29)
     (waiting child469 table300)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child138)
     (served child469)
    )
  )
)
