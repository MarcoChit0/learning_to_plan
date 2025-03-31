; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 664879

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child37 - child
    bread145 - bread-portion
    content73 - content-portion
    tray292 tray39 tray376 tray117 - tray
    table423 table142 table110 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray292 kitchen)
     (at tray39 kitchen)
     (at tray376 kitchen)
     (at tray117 kitchen)
     (at_kitchen_bread bread145)
     (at_kitchen_content content73)
     (not_allergic_gluten child37)
     (waiting child37 table110)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child37)
    )
  )
)
