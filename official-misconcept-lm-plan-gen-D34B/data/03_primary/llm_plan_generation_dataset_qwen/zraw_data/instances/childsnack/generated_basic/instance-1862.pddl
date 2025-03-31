; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 375629

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child168 child101 - child
    bread115 bread304 - bread-portion
    content153 content67 - content-portion
    tray244 tray217 - tray
    table271 table454 table141 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray244 kitchen)
     (at tray217 kitchen)
     (at_kitchen_bread bread115)
     (at_kitchen_bread bread304)
     (at_kitchen_content content153)
     (at_kitchen_content content67)
     (no_gluten_bread bread304)
     (no_gluten_content content67)
     (allergic_gluten child168)
     (not_allergic_gluten child101)
     (waiting child168 table271)
     (waiting child101 table454)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child168)
     (served child101)
    )
  )
)
