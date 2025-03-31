; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 830180

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child423 child360 - child
    bread429 bread94 - bread-portion
    content110 content303 - content-portion
    tray322 tray407 - tray
    table417 table331 table486 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray322 kitchen)
     (at tray407 kitchen)
     (at_kitchen_bread bread429)
     (at_kitchen_bread bread94)
     (at_kitchen_content content110)
     (at_kitchen_content content303)
     (not_allergic_gluten child423)
     (not_allergic_gluten child360)
     (waiting child423 table486)
     (waiting child360 table417)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child423)
     (served child360)
    )
  )
)
