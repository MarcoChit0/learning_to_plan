; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 543686

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child23 - child
    bread10 - bread-portion
    content265 - content-portion
    tray140 tray432 tray20 tray371 - tray
    table465 table3 table138 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray140 kitchen)
     (at tray432 kitchen)
     (at tray20 kitchen)
     (at tray371 kitchen)
     (at_kitchen_bread bread10)
     (at_kitchen_content content265)
     (not_allergic_gluten child23)
     (waiting child23 table3)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child23)
    )
  )
)
