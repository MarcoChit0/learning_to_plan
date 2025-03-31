; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 331148

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child172 child280 - child
    bread171 bread455 - bread-portion
    content314 content362 - content-portion
    tray366 - tray
    table486 table169 table143 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray366 kitchen)
     (at_kitchen_bread bread171)
     (at_kitchen_bread bread455)
     (at_kitchen_content content314)
     (at_kitchen_content content362)
     (no_gluten_bread bread171)
     (no_gluten_content content314)
     (allergic_gluten child280)
     (not_allergic_gluten child172)
     (waiting child172 table143)
     (waiting child280 table169)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child172)
     (served child280)
    )
  )
)
