; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 921630

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child130 child128 - child
    bread398 bread411 - bread-portion
    content48 content499 - content-portion
    tray134 tray450 tray71 tray313 - tray
    table451 table372 table493 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray134 kitchen)
     (at tray450 kitchen)
     (at tray71 kitchen)
     (at tray313 kitchen)
     (at_kitchen_bread bread398)
     (at_kitchen_bread bread411)
     (at_kitchen_content content48)
     (at_kitchen_content content499)
     (no_gluten_bread bread398)
     (no_gluten_content content499)
     (allergic_gluten child128)
     (not_allergic_gluten child130)
     (waiting child130 table372)
     (waiting child128 table451)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child130)
     (served child128)
    )
  )
)
