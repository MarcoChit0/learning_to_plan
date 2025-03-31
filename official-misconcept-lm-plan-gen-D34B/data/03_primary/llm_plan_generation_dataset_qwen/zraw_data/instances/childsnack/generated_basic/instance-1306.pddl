; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.5
; random seed: 538356

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child448 child387 - child
    bread11 bread110 - bread-portion
    content445 content219 - content-portion
    tray409 tray199 tray415 tray0 - tray
    table353 table194 table375 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray409 kitchen)
     (at tray199 kitchen)
     (at tray415 kitchen)
     (at tray0 kitchen)
     (at_kitchen_bread bread11)
     (at_kitchen_bread bread110)
     (at_kitchen_content content445)
     (at_kitchen_content content219)
     (no_gluten_bread bread110)
     (no_gluten_content content219)
     (allergic_gluten child448)
     (not_allergic_gluten child387)
     (waiting child448 table194)
     (waiting child387 table353)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child448)
     (served child387)
    )
  )
)
