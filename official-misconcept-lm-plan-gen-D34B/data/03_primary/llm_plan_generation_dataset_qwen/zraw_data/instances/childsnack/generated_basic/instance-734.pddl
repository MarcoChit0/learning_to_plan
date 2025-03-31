; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 613801

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child13 child196 - child
    bread44 bread348 - bread-portion
    content127 content220 - content-portion
    tray459 tray482 - tray
    table333 table413 table203 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray459 kitchen)
     (at tray482 kitchen)
     (at_kitchen_bread bread44)
     (at_kitchen_bread bread348)
     (at_kitchen_content content127)
     (at_kitchen_content content220)
     (no_gluten_bread bread348)
     (no_gluten_content content127)
     (allergic_gluten child13)
     (not_allergic_gluten child196)
     (waiting child13 table333)
     (waiting child196 table413)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child13)
     (served child196)
    )
  )
)
