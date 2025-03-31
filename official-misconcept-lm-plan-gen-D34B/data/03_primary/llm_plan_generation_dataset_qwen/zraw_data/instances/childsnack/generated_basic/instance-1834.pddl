; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 829516

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child99 child358 - child
    bread70 bread494 - bread-portion
    content349 content98 - content-portion
    tray484 - tray
    table380 table221 table208 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray484 kitchen)
     (at_kitchen_bread bread70)
     (at_kitchen_bread bread494)
     (at_kitchen_content content349)
     (at_kitchen_content content98)
     (no_gluten_bread bread494)
     (no_gluten_content content98)
     (allergic_gluten child358)
     (not_allergic_gluten child99)
     (waiting child99 table380)
     (waiting child358 table380)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child99)
     (served child358)
    )
  )
)
