; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 756304

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child407 - child
    bread485 - bread-portion
    content402 - content-portion
    tray101 tray22 tray12 - tray
    table110 table305 table269 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray101 kitchen)
     (at tray22 kitchen)
     (at tray12 kitchen)
     (at_kitchen_bread bread485)
     (at_kitchen_content content402)
     (not_allergic_gluten child407)
     (waiting child407 table110)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child407)
    )
  )
)
