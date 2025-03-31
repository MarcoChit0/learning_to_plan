; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 436578

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child72 - child
    bread308 - bread-portion
    content219 - content-portion
    tray90 tray177 tray408 tray299 - tray
    table448 table91 table105 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray90 kitchen)
     (at tray177 kitchen)
     (at tray408 kitchen)
     (at tray299 kitchen)
     (at_kitchen_bread bread308)
     (at_kitchen_content content219)
     (not_allergic_gluten child72)
     (waiting child72 table105)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child72)
    )
  )
)
