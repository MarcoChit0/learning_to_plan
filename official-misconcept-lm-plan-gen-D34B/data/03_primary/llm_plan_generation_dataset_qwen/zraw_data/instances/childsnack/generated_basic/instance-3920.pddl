; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 648960

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child84 child104 - child
    bread254 bread227 - bread-portion
    content370 content327 - content-portion
    tray131 - tray
    table47 table35 table3 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray131 kitchen)
     (at_kitchen_bread bread254)
     (at_kitchen_bread bread227)
     (at_kitchen_content content370)
     (at_kitchen_content content327)
     (no_gluten_bread bread227)
     (no_gluten_content content327)
     (allergic_gluten child104)
     (not_allergic_gluten child84)
     (waiting child84 table47)
     (waiting child104 table47)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child84)
     (served child104)
    )
  )
)
