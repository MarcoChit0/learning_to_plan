; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 47519

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child211 child364 - child
    bread49 bread457 - bread-portion
    content257 content115 - content-portion
    tray101 - tray
    table257 table158 table34 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray101 kitchen)
     (at_kitchen_bread bread49)
     (at_kitchen_bread bread457)
     (at_kitchen_content content257)
     (at_kitchen_content content115)
     (no_gluten_bread bread49)
     (no_gluten_content content257)
     (allergic_gluten child211)
     (not_allergic_gluten child364)
     (waiting child211 table257)
     (waiting child364 table34)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child211)
     (served child364)
    )
  )
)
