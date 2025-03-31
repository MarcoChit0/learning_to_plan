; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 256304

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child343 - child
    bread321 - bread-portion
    content461 - content-portion
    tray493 - tray
    table247 table212 table271 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray493 kitchen)
     (at_kitchen_bread bread321)
     (at_kitchen_content content461)
     (not_allergic_gluten child343)
     (waiting child343 table247)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child343)
    )
  )
)
