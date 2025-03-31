; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 518598

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child55 - child
    bread258 - bread-portion
    content93 - content-portion
    tray297 tray114 tray145 - tray
    table233 table283 table352 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray297 kitchen)
     (at tray114 kitchen)
     (at tray145 kitchen)
     (at_kitchen_bread bread258)
     (at_kitchen_content content93)
     (not_allergic_gluten child55)
     (waiting child55 table352)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child55)
    )
  )
)
