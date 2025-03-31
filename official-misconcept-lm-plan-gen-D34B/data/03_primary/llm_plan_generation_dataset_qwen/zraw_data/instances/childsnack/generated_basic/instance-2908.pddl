; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 56703

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child267 child126 - child
    bread31 bread331 - bread-portion
    content35 content167 - content-portion
    tray481 tray292 tray79 - tray
    table74 table348 table391 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray481 kitchen)
     (at tray292 kitchen)
     (at tray79 kitchen)
     (at_kitchen_bread bread31)
     (at_kitchen_bread bread331)
     (at_kitchen_content content35)
     (at_kitchen_content content167)
     (no_gluten_bread bread331)
     (no_gluten_content content35)
     (allergic_gluten child267)
     (not_allergic_gluten child126)
     (waiting child267 table348)
     (waiting child126 table391)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child267)
     (served child126)
    )
  )
)
