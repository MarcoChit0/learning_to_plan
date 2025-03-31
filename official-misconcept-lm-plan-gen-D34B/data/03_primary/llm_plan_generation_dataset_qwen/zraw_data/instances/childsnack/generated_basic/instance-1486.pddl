; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 562575

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child7 child169 - child
    bread3 bread407 - bread-portion
    content212 content189 - content-portion
    tray213 tray69 tray18 - tray
    table155 table107 table222 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray213 kitchen)
     (at tray69 kitchen)
     (at tray18 kitchen)
     (at_kitchen_bread bread3)
     (at_kitchen_bread bread407)
     (at_kitchen_content content212)
     (at_kitchen_content content189)
     (no_gluten_bread bread407)
     (no_gluten_content content212)
     (allergic_gluten child169)
     (not_allergic_gluten child7)
     (waiting child7 table107)
     (waiting child169 table222)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child7)
     (served child169)
    )
  )
)
