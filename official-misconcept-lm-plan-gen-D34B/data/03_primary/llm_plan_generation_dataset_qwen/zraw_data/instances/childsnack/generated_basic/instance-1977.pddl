; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 534446

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child309 child180 - child
    bread298 bread199 - bread-portion
    content427 content245 - content-portion
    tray351 - tray
    table52 table343 table420 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray351 kitchen)
     (at_kitchen_bread bread298)
     (at_kitchen_bread bread199)
     (at_kitchen_content content427)
     (at_kitchen_content content245)
     (no_gluten_bread bread298)
     (no_gluten_content content245)
     (allergic_gluten child180)
     (not_allergic_gluten child309)
     (waiting child309 table343)
     (waiting child180 table420)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child309)
     (served child180)
    )
  )
)
