; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 257298

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child344 child167 - child
    bread387 bread102 - bread-portion
    content119 content188 - content-portion
    tray387 - tray
    table38 table120 table493 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray387 kitchen)
     (at_kitchen_bread bread387)
     (at_kitchen_bread bread102)
     (at_kitchen_content content119)
     (at_kitchen_content content188)
     (not_allergic_gluten child344)
     (not_allergic_gluten child167)
     (waiting child344 table493)
     (waiting child167 table38)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child344)
     (served child167)
    )
  )
)
