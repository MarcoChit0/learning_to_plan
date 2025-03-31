; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 572413

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child412 - child
    bread449 - bread-portion
    content143 - content-portion
    tray291 tray432 tray350 tray464 - tray
    table197 table452 table6 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray291 kitchen)
     (at tray432 kitchen)
     (at tray350 kitchen)
     (at tray464 kitchen)
     (at_kitchen_bread bread449)
     (at_kitchen_content content143)
     (not_allergic_gluten child412)
     (waiting child412 table197)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child412)
    )
  )
)
