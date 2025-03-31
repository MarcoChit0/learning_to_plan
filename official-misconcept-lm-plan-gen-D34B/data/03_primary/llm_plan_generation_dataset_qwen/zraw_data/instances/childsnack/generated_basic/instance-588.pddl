; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 835746

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child4 - child
    bread200 - bread-portion
    content456 - content-portion
    tray191 - tray
    table40 table21 table473 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray191 kitchen)
     (at_kitchen_bread bread200)
     (at_kitchen_content content456)
     (not_allergic_gluten child4)
     (waiting child4 table40)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child4)
    )
  )
)
