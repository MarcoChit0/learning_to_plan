; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 190534

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child69 - child
    bread499 - bread-portion
    content230 - content-portion
    tray366 tray56 tray330 - tray
    table458 table117 table214 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray366 kitchen)
     (at tray56 kitchen)
     (at tray330 kitchen)
     (at_kitchen_bread bread499)
     (at_kitchen_content content230)
     (not_allergic_gluten child69)
     (waiting child69 table117)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child69)
    )
  )
)
