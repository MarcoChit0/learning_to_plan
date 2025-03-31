; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 467453

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child62 child439 - child
    bread9 bread47 - bread-portion
    content401 content182 - content-portion
    tray360 - tray
    table259 table402 table14 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray360 kitchen)
     (at_kitchen_bread bread9)
     (at_kitchen_bread bread47)
     (at_kitchen_content content401)
     (at_kitchen_content content182)
     (not_allergic_gluten child62)
     (not_allergic_gluten child439)
     (waiting child62 table402)
     (waiting child439 table259)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child62)
     (served child439)
    )
  )
)
