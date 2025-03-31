; child-snack task with 2 children and 0.6 gluten factor 
; constant factor of 1.3
; random seed: 231991

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child296 child130 - child
    bread234 bread270 - bread-portion
    content285 content474 - content-portion
    tray212 tray413 - tray
    table286 table122 table164 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray212 kitchen)
     (at tray413 kitchen)
     (at_kitchen_bread bread234)
     (at_kitchen_bread bread270)
     (at_kitchen_content content285)
     (at_kitchen_content content474)
     (no_gluten_bread bread234)
     (no_gluten_content content285)
     (allergic_gluten child130)
     (not_allergic_gluten child296)
     (waiting child296 table164)
     (waiting child130 table122)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child296)
     (served child130)
    )
  )
)
