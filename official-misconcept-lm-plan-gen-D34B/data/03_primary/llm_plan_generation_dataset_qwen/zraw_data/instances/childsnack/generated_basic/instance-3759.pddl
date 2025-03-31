; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 499386

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child372 - child
    bread204 - bread-portion
    content454 - content-portion
    tray235 tray470 tray3 - tray
    table218 table164 table343 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray235 kitchen)
     (at tray470 kitchen)
     (at tray3 kitchen)
     (at_kitchen_bread bread204)
     (at_kitchen_content content454)
     (not_allergic_gluten child372)
     (waiting child372 table218)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child372)
    )
  )
)
