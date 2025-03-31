; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 993607

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child385 - child
    bread441 - bread-portion
    content407 - content-portion
    tray385 tray394 tray370 tray433 - tray
    table419 table346 table179 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray385 kitchen)
     (at tray394 kitchen)
     (at tray370 kitchen)
     (at tray433 kitchen)
     (at_kitchen_bread bread441)
     (at_kitchen_content content407)
     (not_allergic_gluten child385)
     (waiting child385 table419)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child385)
    )
  )
)
