; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 582490

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child0 child476 child496 - child
    bread321 bread489 bread37 - bread-portion
    content49 content376 content47 - content-portion
    tray76 - tray
    table139 table88 table64 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray76 kitchen)
     (at_kitchen_bread bread321)
     (at_kitchen_bread bread489)
     (at_kitchen_bread bread37)
     (at_kitchen_content content49)
     (at_kitchen_content content376)
     (at_kitchen_content content47)
     (not_allergic_gluten child476)
     (not_allergic_gluten child0)
     (not_allergic_gluten child496)
     (waiting child0 table88)
     (waiting child476 table64)
     (waiting child496 table139)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child0)
     (served child476)
     (served child496)
    )
  )
)
