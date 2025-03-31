; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 635682

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child441 - child
    bread496 - bread-portion
    content456 - content-portion
    tray462 tray455 tray392 - tray
    table425 table485 table190 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray462 kitchen)
     (at tray455 kitchen)
     (at tray392 kitchen)
     (at_kitchen_bread bread496)
     (at_kitchen_content content456)
     (not_allergic_gluten child441)
     (waiting child441 table190)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child441)
    )
  )
)
