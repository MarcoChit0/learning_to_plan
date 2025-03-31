; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 12237

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child163 - child
    bread354 - bread-portion
    content430 - content-portion
    tray356 tray443 tray486 tray15 - tray
    table313 table343 table364 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray356 kitchen)
     (at tray443 kitchen)
     (at tray486 kitchen)
     (at tray15 kitchen)
     (at_kitchen_bread bread354)
     (at_kitchen_content content430)
     (not_allergic_gluten child163)
     (waiting child163 table313)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child163)
    )
  )
)
