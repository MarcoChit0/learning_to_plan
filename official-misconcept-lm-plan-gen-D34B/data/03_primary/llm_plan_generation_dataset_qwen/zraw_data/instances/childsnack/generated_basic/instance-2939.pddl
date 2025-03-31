; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 142305

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child245 child433 - child
    bread489 bread341 - bread-portion
    content300 content336 - content-portion
    tray217 tray123 tray218 - tray
    table182 table272 table299 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray217 kitchen)
     (at tray123 kitchen)
     (at tray218 kitchen)
     (at_kitchen_bread bread489)
     (at_kitchen_bread bread341)
     (at_kitchen_content content300)
     (at_kitchen_content content336)
     (no_gluten_bread bread341)
     (no_gluten_content content300)
     (allergic_gluten child245)
     (not_allergic_gluten child433)
     (waiting child245 table272)
     (waiting child433 table299)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child245)
     (served child433)
    )
  )
)
