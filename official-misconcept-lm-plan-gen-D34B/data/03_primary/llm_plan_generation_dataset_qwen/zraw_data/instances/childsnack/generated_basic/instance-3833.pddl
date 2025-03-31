; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 4406

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child365 - child
    bread367 - bread-portion
    content333 - content-portion
    tray378 tray451 tray63 tray33 - tray
    table399 table396 table54 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray378 kitchen)
     (at tray451 kitchen)
     (at tray63 kitchen)
     (at tray33 kitchen)
     (at_kitchen_bread bread367)
     (at_kitchen_content content333)
     (not_allergic_gluten child365)
     (waiting child365 table54)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child365)
    )
  )
)
