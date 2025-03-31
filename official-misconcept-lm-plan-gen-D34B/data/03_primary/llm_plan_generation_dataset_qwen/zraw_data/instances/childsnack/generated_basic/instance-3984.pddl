; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 512530

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child433 - child
    bread305 - bread-portion
    content109 - content-portion
    tray359 tray128 tray276 - tray
    table319 table256 table403 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray359 kitchen)
     (at tray128 kitchen)
     (at tray276 kitchen)
     (at_kitchen_bread bread305)
     (at_kitchen_content content109)
     (not_allergic_gluten child433)
     (waiting child433 table319)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child433)
    )
  )
)
