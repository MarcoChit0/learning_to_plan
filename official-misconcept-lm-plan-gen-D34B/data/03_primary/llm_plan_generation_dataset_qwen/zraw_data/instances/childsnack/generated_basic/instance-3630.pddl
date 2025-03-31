; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 396879

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child155 child187 - child
    bread237 bread386 - bread-portion
    content129 content284 - content-portion
    tray168 tray54 tray420 tray66 - tray
    table380 table63 table348 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray168 kitchen)
     (at tray54 kitchen)
     (at tray420 kitchen)
     (at tray66 kitchen)
     (at_kitchen_bread bread237)
     (at_kitchen_bread bread386)
     (at_kitchen_content content129)
     (at_kitchen_content content284)
     (no_gluten_bread bread386)
     (no_gluten_content content284)
     (allergic_gluten child187)
     (not_allergic_gluten child155)
     (waiting child155 table63)
     (waiting child187 table63)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child155)
     (served child187)
    )
  )
)
