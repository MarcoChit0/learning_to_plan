; child-snack task with 2 children and 0.56 gluten factor 
; constant factor of 1.3
; random seed: 78828

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child190 child284 - child
    bread314 bread310 - bread-portion
    content490 content104 - content-portion
    tray34 tray78 - tray
    table427 table365 table246 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray34 kitchen)
     (at tray78 kitchen)
     (at_kitchen_bread bread314)
     (at_kitchen_bread bread310)
     (at_kitchen_content content490)
     (at_kitchen_content content104)
     (no_gluten_bread bread314)
     (no_gluten_content content104)
     (allergic_gluten child190)
     (not_allergic_gluten child284)
     (waiting child190 table365)
     (waiting child284 table246)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child190)
     (served child284)
    )
  )
)
