; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 587073

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child60 - child
    bread240 - bread-portion
    content388 - content-portion
    tray45 tray202 tray369 tray201 - tray
    table305 table491 table373 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray45 kitchen)
     (at tray202 kitchen)
     (at tray369 kitchen)
     (at tray201 kitchen)
     (at_kitchen_bread bread240)
     (at_kitchen_content content388)
     (not_allergic_gluten child60)
     (waiting child60 table373)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child60)
    )
  )
)
