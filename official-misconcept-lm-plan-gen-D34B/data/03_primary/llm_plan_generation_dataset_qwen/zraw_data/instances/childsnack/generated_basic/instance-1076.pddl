; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 476548

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child201 child125 - child
    bread31 bread28 - bread-portion
    content29 content177 - content-portion
    tray369 tray472 tray269 - tray
    table98 table382 table58 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray369 kitchen)
     (at tray472 kitchen)
     (at tray269 kitchen)
     (at_kitchen_bread bread31)
     (at_kitchen_bread bread28)
     (at_kitchen_content content29)
     (at_kitchen_content content177)
     (not_allergic_gluten child201)
     (not_allergic_gluten child125)
     (waiting child201 table58)
     (waiting child125 table98)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child201)
     (served child125)
    )
  )
)
