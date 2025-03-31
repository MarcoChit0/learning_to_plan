; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 780048

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child127 - child
    bread128 - bread-portion
    content430 - content-portion
    tray340 tray127 tray247 tray170 - tray
    table339 table116 table498 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray340 kitchen)
     (at tray127 kitchen)
     (at tray247 kitchen)
     (at tray170 kitchen)
     (at_kitchen_bread bread128)
     (at_kitchen_content content430)
     (not_allergic_gluten child127)
     (waiting child127 table498)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child127)
    )
  )
)
