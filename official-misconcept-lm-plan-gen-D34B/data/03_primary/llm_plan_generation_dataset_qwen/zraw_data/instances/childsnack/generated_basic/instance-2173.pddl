; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 445126

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child415 - child
    bread86 - bread-portion
    content180 - content-portion
    tray324 - tray
    table320 table175 table15 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray324 kitchen)
     (at_kitchen_bread bread86)
     (at_kitchen_content content180)
     (not_allergic_gluten child415)
     (waiting child415 table320)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child415)
    )
  )
)
