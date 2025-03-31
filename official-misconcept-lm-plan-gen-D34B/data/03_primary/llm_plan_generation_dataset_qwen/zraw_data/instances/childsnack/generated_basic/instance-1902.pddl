; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 354251

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child338 child252 - child
    bread22 bread469 - bread-portion
    content71 content254 - content-portion
    tray265 tray73 - tray
    table449 table62 table342 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray265 kitchen)
     (at tray73 kitchen)
     (at_kitchen_bread bread22)
     (at_kitchen_bread bread469)
     (at_kitchen_content content71)
     (at_kitchen_content content254)
     (no_gluten_bread bread469)
     (no_gluten_content content254)
     (allergic_gluten child338)
     (not_allergic_gluten child252)
     (waiting child338 table342)
     (waiting child252 table62)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child338)
     (served child252)
    )
  )
)
