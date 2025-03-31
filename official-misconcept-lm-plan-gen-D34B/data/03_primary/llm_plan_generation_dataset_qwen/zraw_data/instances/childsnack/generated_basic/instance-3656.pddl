; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 10338

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child96 child282 - child
    bread409 bread344 - bread-portion
    content254 content471 - content-portion
    tray226 tray420 tray464 tray459 - tray
    table64 table460 table99 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray226 kitchen)
     (at tray420 kitchen)
     (at tray464 kitchen)
     (at tray459 kitchen)
     (at_kitchen_bread bread409)
     (at_kitchen_bread bread344)
     (at_kitchen_content content254)
     (at_kitchen_content content471)
     (no_gluten_bread bread409)
     (no_gluten_content content254)
     (allergic_gluten child96)
     (not_allergic_gluten child282)
     (waiting child96 table460)
     (waiting child282 table99)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child96)
     (served child282)
    )
  )
)
