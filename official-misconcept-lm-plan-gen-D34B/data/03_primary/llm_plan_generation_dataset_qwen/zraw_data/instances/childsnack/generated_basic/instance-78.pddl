; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 579207

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child56 child149 - child
    bread80 bread86 - bread-portion
    content262 content95 - content-portion
    tray454 tray478 - tray
    table451 table222 table481 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray454 kitchen)
     (at tray478 kitchen)
     (at_kitchen_bread bread80)
     (at_kitchen_bread bread86)
     (at_kitchen_content content262)
     (at_kitchen_content content95)
     (not_allergic_gluten child149)
     (not_allergic_gluten child56)
     (waiting child56 table451)
     (waiting child149 table222)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child56)
     (served child149)
    )
  )
)
