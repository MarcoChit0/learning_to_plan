; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 388373

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child21 child415 - child
    bread221 bread382 - bread-portion
    content219 content148 - content-portion
    tray431 tray113 - tray
    table438 table74 table317 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray431 kitchen)
     (at tray113 kitchen)
     (at_kitchen_bread bread221)
     (at_kitchen_bread bread382)
     (at_kitchen_content content219)
     (at_kitchen_content content148)
     (not_allergic_gluten child415)
     (not_allergic_gluten child21)
     (waiting child21 table74)
     (waiting child415 table74)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child21)
     (served child415)
    )
  )
)
