; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 952390

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child28 child209 - child
    bread398 bread486 - bread-portion
    content109 content484 - content-portion
    tray437 - tray
    table493 table234 table22 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray437 kitchen)
     (at_kitchen_bread bread398)
     (at_kitchen_bread bread486)
     (at_kitchen_content content109)
     (at_kitchen_content content484)
     (no_gluten_bread bread398)
     (no_gluten_content content109)
     (allergic_gluten child209)
     (not_allergic_gluten child28)
     (waiting child28 table493)
     (waiting child209 table493)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child28)
     (served child209)
    )
  )
)
