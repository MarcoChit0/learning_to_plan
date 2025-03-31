; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 377901

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child394 child473 - child
    bread499 bread327 - bread-portion
    content373 content375 - content-portion
    tray154 tray42 - tray
    table400 table295 table290 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray154 kitchen)
     (at tray42 kitchen)
     (at_kitchen_bread bread499)
     (at_kitchen_bread bread327)
     (at_kitchen_content content373)
     (at_kitchen_content content375)
     (no_gluten_bread bread499)
     (no_gluten_content content373)
     (allergic_gluten child394)
     (not_allergic_gluten child473)
     (waiting child394 table295)
     (waiting child473 table290)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child394)
     (served child473)
    )
  )
)
