; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 310357

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child146 child232 - child
    bread439 bread483 - bread-portion
    content304 content488 - content-portion
    tray191 tray38 - tray
    table453 table473 table497 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray191 kitchen)
     (at tray38 kitchen)
     (at_kitchen_bread bread439)
     (at_kitchen_bread bread483)
     (at_kitchen_content content304)
     (at_kitchen_content content488)
     (not_allergic_gluten child146)
     (not_allergic_gluten child232)
     (waiting child146 table453)
     (waiting child232 table497)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child146)
     (served child232)
    )
  )
)
