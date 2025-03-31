; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 941336

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child201 child265 - child
    bread92 bread72 - bread-portion
    content108 content208 - content-portion
    tray384 - tray
    table375 table493 table202 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray384 kitchen)
     (at_kitchen_bread bread92)
     (at_kitchen_bread bread72)
     (at_kitchen_content content108)
     (at_kitchen_content content208)
     (not_allergic_gluten child201)
     (not_allergic_gluten child265)
     (waiting child201 table202)
     (waiting child265 table493)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child201)
     (served child265)
    )
  )
)
