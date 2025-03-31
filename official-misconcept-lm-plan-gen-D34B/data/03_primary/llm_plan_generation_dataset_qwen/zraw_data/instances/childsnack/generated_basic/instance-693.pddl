; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 962812

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child372 - child
    bread272 - bread-portion
    content490 - content-portion
    tray308 tray33 tray485 tray380 - tray
    table215 table28 table158 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray308 kitchen)
     (at tray33 kitchen)
     (at tray485 kitchen)
     (at tray380 kitchen)
     (at_kitchen_bread bread272)
     (at_kitchen_content content490)
     (not_allergic_gluten child372)
     (waiting child372 table28)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child372)
    )
  )
)
