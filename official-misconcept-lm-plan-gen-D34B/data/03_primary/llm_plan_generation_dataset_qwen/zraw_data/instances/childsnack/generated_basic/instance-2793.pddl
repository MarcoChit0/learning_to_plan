; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 564971

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child100 child455 - child
    bread337 bread479 - bread-portion
    content52 content113 - content-portion
    tray15 tray305 - tray
    table148 table314 table466 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray15 kitchen)
     (at tray305 kitchen)
     (at_kitchen_bread bread337)
     (at_kitchen_bread bread479)
     (at_kitchen_content content52)
     (at_kitchen_content content113)
     (not_allergic_gluten child455)
     (not_allergic_gluten child100)
     (waiting child100 table466)
     (waiting child455 table466)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child100)
     (served child455)
    )
  )
)
