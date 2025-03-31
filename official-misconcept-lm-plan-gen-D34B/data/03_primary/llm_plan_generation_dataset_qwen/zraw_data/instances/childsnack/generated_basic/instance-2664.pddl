; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 707122

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child244 - child
    bread148 - bread-portion
    content98 - content-portion
    tray238 tray189 tray370 tray307 - tray
    table122 table142 table22 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray238 kitchen)
     (at tray189 kitchen)
     (at tray370 kitchen)
     (at tray307 kitchen)
     (at_kitchen_bread bread148)
     (at_kitchen_content content98)
     (not_allergic_gluten child244)
     (waiting child244 table22)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child244)
    )
  )
)
