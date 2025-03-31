; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 297944

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child258 - child
    bread271 - bread-portion
    content332 - content-portion
    tray147 tray224 tray417 - tray
    table227 table202 table28 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray147 kitchen)
     (at tray224 kitchen)
     (at tray417 kitchen)
     (at_kitchen_bread bread271)
     (at_kitchen_content content332)
     (not_allergic_gluten child258)
     (waiting child258 table227)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child258)
    )
  )
)
