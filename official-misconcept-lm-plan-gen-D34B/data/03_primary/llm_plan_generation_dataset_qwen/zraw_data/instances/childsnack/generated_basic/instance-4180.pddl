; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 324977

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child361 child238 - child
    bread487 bread477 - bread-portion
    content462 content226 - content-portion
    tray189 - tray
    table128 table403 table73 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray189 kitchen)
     (at_kitchen_bread bread487)
     (at_kitchen_bread bread477)
     (at_kitchen_content content462)
     (at_kitchen_content content226)
     (not_allergic_gluten child361)
     (not_allergic_gluten child238)
     (waiting child361 table403)
     (waiting child238 table73)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child361)
     (served child238)
    )
  )
)
