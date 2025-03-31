; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 910093

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child492 child459 - child
    bread45 bread201 - bread-portion
    content410 content134 - content-portion
    tray274 - tray
    table391 table488 table297 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray274 kitchen)
     (at_kitchen_bread bread45)
     (at_kitchen_bread bread201)
     (at_kitchen_content content410)
     (at_kitchen_content content134)
     (not_allergic_gluten child492)
     (not_allergic_gluten child459)
     (waiting child492 table488)
     (waiting child459 table297)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child492)
     (served child459)
    )
  )
)
