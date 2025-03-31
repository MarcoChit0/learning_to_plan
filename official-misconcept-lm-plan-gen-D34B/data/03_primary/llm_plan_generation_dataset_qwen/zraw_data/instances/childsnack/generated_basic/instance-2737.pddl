; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 610924

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child187 child189 - child
    bread188 bread285 - bread-portion
    content465 content55 - content-portion
    tray123 - tray
    table80 table164 table354 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray123 kitchen)
     (at_kitchen_bread bread188)
     (at_kitchen_bread bread285)
     (at_kitchen_content content465)
     (at_kitchen_content content55)
     (no_gluten_bread bread285)
     (no_gluten_content content465)
     (allergic_gluten child189)
     (not_allergic_gluten child187)
     (waiting child187 table354)
     (waiting child189 table354)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child187)
     (served child189)
    )
  )
)
