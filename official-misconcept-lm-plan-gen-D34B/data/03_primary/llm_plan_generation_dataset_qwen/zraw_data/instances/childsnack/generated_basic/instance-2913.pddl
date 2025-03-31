; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 945262

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child474 child490 - child
    bread266 bread77 - bread-portion
    content369 content232 - content-portion
    tray48 tray136 tray100 - tray
    table23 table371 table41 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray48 kitchen)
     (at tray136 kitchen)
     (at tray100 kitchen)
     (at_kitchen_bread bread266)
     (at_kitchen_bread bread77)
     (at_kitchen_content content369)
     (at_kitchen_content content232)
     (no_gluten_bread bread266)
     (no_gluten_content content369)
     (allergic_gluten child490)
     (not_allergic_gluten child474)
     (waiting child474 table41)
     (waiting child490 table41)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child474)
     (served child490)
    )
  )
)
