; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 793328

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child485 child364 - child
    bread30 bread134 - bread-portion
    content425 content256 - content-portion
    tray48 tray222 tray463 tray286 - tray
    table251 table218 table83 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray48 kitchen)
     (at tray222 kitchen)
     (at tray463 kitchen)
     (at tray286 kitchen)
     (at_kitchen_bread bread30)
     (at_kitchen_bread bread134)
     (at_kitchen_content content425)
     (at_kitchen_content content256)
     (no_gluten_bread bread30)
     (no_gluten_content content425)
     (allergic_gluten child364)
     (not_allergic_gluten child485)
     (waiting child485 table218)
     (waiting child364 table218)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child485)
     (served child364)
    )
  )
)
