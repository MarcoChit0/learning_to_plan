; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 945024

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child120 - child
    bread485 - bread-portion
    content104 - content-portion
    tray383 - tray
    table51 table439 table184 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray383 kitchen)
     (at_kitchen_bread bread485)
     (at_kitchen_content content104)
     (not_allergic_gluten child120)
     (waiting child120 table51)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child120)
    )
  )
)
