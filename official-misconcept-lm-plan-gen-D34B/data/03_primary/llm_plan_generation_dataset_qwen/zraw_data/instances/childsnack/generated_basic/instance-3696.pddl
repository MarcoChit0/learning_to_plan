; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 263169

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child169 child161 - child
    bread359 bread490 - bread-portion
    content97 content26 - content-portion
    tray241 tray171 - tray
    table182 table297 table155 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray241 kitchen)
     (at tray171 kitchen)
     (at_kitchen_bread bread359)
     (at_kitchen_bread bread490)
     (at_kitchen_content content97)
     (at_kitchen_content content26)
     (no_gluten_bread bread490)
     (no_gluten_content content26)
     (allergic_gluten child169)
     (not_allergic_gluten child161)
     (waiting child169 table155)
     (waiting child161 table155)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child169)
     (served child161)
    )
  )
)
