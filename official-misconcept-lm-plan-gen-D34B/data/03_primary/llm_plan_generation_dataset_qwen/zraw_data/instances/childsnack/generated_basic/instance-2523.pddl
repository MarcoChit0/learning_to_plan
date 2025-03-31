; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 826499

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child31 - child
    bread331 - bread-portion
    content460 - content-portion
    tray49 tray484 tray84 - tray
    table223 table297 table354 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray49 kitchen)
     (at tray484 kitchen)
     (at tray84 kitchen)
     (at_kitchen_bread bread331)
     (at_kitchen_content content460)
     (not_allergic_gluten child31)
     (waiting child31 table297)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child31)
    )
  )
)
