; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 79651

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child220 child343 - child
    bread378 bread100 - bread-portion
    content327 content318 - content-portion
    tray487 tray346 tray288 - tray
    table261 table389 table121 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray487 kitchen)
     (at tray346 kitchen)
     (at tray288 kitchen)
     (at_kitchen_bread bread378)
     (at_kitchen_bread bread100)
     (at_kitchen_content content327)
     (at_kitchen_content content318)
     (not_allergic_gluten child343)
     (not_allergic_gluten child220)
     (waiting child220 table261)
     (waiting child343 table121)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child220)
     (served child343)
    )
  )
)
