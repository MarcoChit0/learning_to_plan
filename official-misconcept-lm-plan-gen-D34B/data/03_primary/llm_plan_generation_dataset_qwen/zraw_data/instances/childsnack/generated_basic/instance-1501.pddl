; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 280108

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child131 child473 - child
    bread318 bread171 - bread-portion
    content178 content483 - content-portion
    tray176 tray477 tray449 - tray
    table427 table250 table292 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray176 kitchen)
     (at tray477 kitchen)
     (at tray449 kitchen)
     (at_kitchen_bread bread318)
     (at_kitchen_bread bread171)
     (at_kitchen_content content178)
     (at_kitchen_content content483)
     (no_gluten_bread bread171)
     (no_gluten_content content178)
     (allergic_gluten child131)
     (not_allergic_gluten child473)
     (waiting child131 table427)
     (waiting child473 table292)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child131)
     (served child473)
    )
  )
)
