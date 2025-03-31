; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 156320

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child435 child73 - child
    bread248 bread385 - bread-portion
    content248 content124 - content-portion
    tray144 tray265 tray325 - tray
    table477 table468 table232 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray144 kitchen)
     (at tray265 kitchen)
     (at tray325 kitchen)
     (at_kitchen_bread bread248)
     (at_kitchen_bread bread385)
     (at_kitchen_content content248)
     (at_kitchen_content content124)
     (no_gluten_bread bread385)
     (no_gluten_content content248)
     (allergic_gluten child435)
     (not_allergic_gluten child73)
     (waiting child435 table232)
     (waiting child73 table232)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child435)
     (served child73)
    )
  )
)
