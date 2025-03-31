; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 286474

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child351 - child
    bread303 - bread-portion
    content185 - content-portion
    tray250 - tray
    table79 table55 table197 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray250 kitchen)
     (at_kitchen_bread bread303)
     (at_kitchen_content content185)
     (not_allergic_gluten child351)
     (waiting child351 table55)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child351)
    )
  )
)
