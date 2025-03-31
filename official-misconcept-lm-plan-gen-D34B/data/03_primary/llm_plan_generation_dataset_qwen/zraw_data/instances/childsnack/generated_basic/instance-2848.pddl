; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 380385

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child134 - child
    bread487 - bread-portion
    content160 - content-portion
    tray198 tray182 - tray
    table135 table237 table401 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray198 kitchen)
     (at tray182 kitchen)
     (at_kitchen_bread bread487)
     (at_kitchen_content content160)
     (not_allergic_gluten child134)
     (waiting child134 table237)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child134)
    )
  )
)
