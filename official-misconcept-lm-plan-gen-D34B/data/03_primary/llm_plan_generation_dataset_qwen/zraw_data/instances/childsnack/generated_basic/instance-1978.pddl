; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 172300

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child338 child201 - child
    bread429 bread278 - bread-portion
    content64 content193 - content-portion
    tray480 - tray
    table55 table0 table54 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray480 kitchen)
     (at_kitchen_bread bread429)
     (at_kitchen_bread bread278)
     (at_kitchen_content content64)
     (at_kitchen_content content193)
     (no_gluten_bread bread278)
     (no_gluten_content content193)
     (allergic_gluten child201)
     (not_allergic_gluten child338)
     (waiting child338 table0)
     (waiting child201 table54)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child338)
     (served child201)
    )
  )
)
