; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 948204

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child33 - child
    bread288 - bread-portion
    content184 - content-portion
    tray20 tray262 - tray
    table180 table284 table411 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray20 kitchen)
     (at tray262 kitchen)
     (at_kitchen_bread bread288)
     (at_kitchen_content content184)
     (not_allergic_gluten child33)
     (waiting child33 table284)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child33)
    )
  )
)
