; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 946084

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child113 child199 - child
    bread320 bread10 - bread-portion
    content354 content175 - content-portion
    tray471 tray470 tray36 - tray
    table218 table443 table216 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray471 kitchen)
     (at tray470 kitchen)
     (at tray36 kitchen)
     (at_kitchen_bread bread320)
     (at_kitchen_bread bread10)
     (at_kitchen_content content354)
     (at_kitchen_content content175)
     (no_gluten_bread bread320)
     (no_gluten_content content175)
     (allergic_gluten child199)
     (not_allergic_gluten child113)
     (waiting child113 table443)
     (waiting child199 table218)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child113)
     (served child199)
    )
  )
)
