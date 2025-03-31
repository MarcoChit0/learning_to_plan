; child-snack task with 2 children and 0.6 gluten factor 
; constant factor of 1.3
; random seed: 755928

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child456 child445 - child
    bread95 bread214 - bread-portion
    content405 content238 - content-portion
    tray188 tray296 - tray
    table29 table79 table134 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray188 kitchen)
     (at tray296 kitchen)
     (at_kitchen_bread bread95)
     (at_kitchen_bread bread214)
     (at_kitchen_content content405)
     (at_kitchen_content content238)
     (no_gluten_bread bread95)
     (no_gluten_content content238)
     (allergic_gluten child456)
     (not_allergic_gluten child445)
     (waiting child456 table134)
     (waiting child445 table29)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child456)
     (served child445)
    )
  )
)
