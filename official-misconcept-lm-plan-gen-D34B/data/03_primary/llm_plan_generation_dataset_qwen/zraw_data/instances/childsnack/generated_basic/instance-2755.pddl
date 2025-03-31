; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 972816

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child68 child340 - child
    bread374 bread355 - bread-portion
    content131 content208 - content-portion
    tray253 - tray
    table110 table30 table371 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray253 kitchen)
     (at_kitchen_bread bread374)
     (at_kitchen_bread bread355)
     (at_kitchen_content content131)
     (at_kitchen_content content208)
     (no_gluten_bread bread374)
     (no_gluten_content content208)
     (allergic_gluten child340)
     (not_allergic_gluten child68)
     (waiting child68 table110)
     (waiting child340 table110)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child68)
     (served child340)
    )
  )
)
