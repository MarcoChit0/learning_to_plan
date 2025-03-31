; child-snack task with 2 children and 0.65 gluten factor 
; constant factor of 1.3
; random seed: 423333

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child385 child387 - child
    bread298 bread259 - bread-portion
    content179 content445 - content-portion
    tray233 tray422 tray349 tray228 - tray
    table185 table38 table475 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray233 kitchen)
     (at tray422 kitchen)
     (at tray349 kitchen)
     (at tray228 kitchen)
     (at_kitchen_bread bread298)
     (at_kitchen_bread bread259)
     (at_kitchen_content content179)
     (at_kitchen_content content445)
     (no_gluten_bread bread298)
     (no_gluten_content content445)
     (allergic_gluten child387)
     (not_allergic_gluten child385)
     (waiting child385 table38)
     (waiting child387 table475)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child385)
     (served child387)
    )
  )
)
