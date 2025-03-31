; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 433795

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child343 - child
    bread370 - bread-portion
    content71 - content-portion
    tray335 - tray
    table99 table386 table260 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray335 kitchen)
     (at_kitchen_bread bread370)
     (at_kitchen_content content71)
     (not_allergic_gluten child343)
     (waiting child343 table99)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child343)
    )
  )
)
