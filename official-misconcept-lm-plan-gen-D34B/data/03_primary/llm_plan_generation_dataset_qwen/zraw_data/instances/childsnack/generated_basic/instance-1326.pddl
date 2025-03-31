; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.5
; random seed: 223808

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child448 child133 - child
    bread486 bread441 - bread-portion
    content140 content83 - content-portion
    tray499 tray434 tray37 tray380 - tray
    table116 table2 table351 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray499 kitchen)
     (at tray434 kitchen)
     (at tray37 kitchen)
     (at tray380 kitchen)
     (at_kitchen_bread bread486)
     (at_kitchen_bread bread441)
     (at_kitchen_content content140)
     (at_kitchen_content content83)
     (no_gluten_bread bread441)
     (no_gluten_content content140)
     (allergic_gluten child448)
     (not_allergic_gluten child133)
     (waiting child448 table2)
     (waiting child133 table116)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child448)
     (served child133)
    )
  )
)
