; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 656567

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child490 - child
    bread2 - bread-portion
    content110 - content-portion
    tray54 tray369 - tray
    table22 table153 table419 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray54 kitchen)
     (at tray369 kitchen)
     (at_kitchen_bread bread2)
     (at_kitchen_content content110)
     (not_allergic_gluten child490)
     (waiting child490 table419)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child490)
    )
  )
)
