; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 421872

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child279 child53 child170 - child
    bread79 bread421 bread119 - bread-portion
    content247 content387 content309 - content-portion
    tray365 - tray
    table433 table435 table28 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray365 kitchen)
     (at_kitchen_bread bread79)
     (at_kitchen_bread bread421)
     (at_kitchen_bread bread119)
     (at_kitchen_content content247)
     (at_kitchen_content content387)
     (at_kitchen_content content309)
     (not_allergic_gluten child279)
     (not_allergic_gluten child53)
     (not_allergic_gluten child170)
     (waiting child279 table435)
     (waiting child53 table435)
     (waiting child170 table28)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child279)
     (served child53)
     (served child170)
    )
  )
)
