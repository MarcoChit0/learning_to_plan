; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 347897

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child39 child41 - child
    bread24 bread37 - bread-portion
    content454 content173 - content-portion
    tray197 tray489 tray226 - tray
    table166 table111 table62 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray197 kitchen)
     (at tray489 kitchen)
     (at tray226 kitchen)
     (at_kitchen_bread bread24)
     (at_kitchen_bread bread37)
     (at_kitchen_content content454)
     (at_kitchen_content content173)
     (not_allergic_gluten child39)
     (not_allergic_gluten child41)
     (waiting child39 table62)
     (waiting child41 table111)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child39)
     (served child41)
    )
  )
)
