; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 951195

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child120 child89 - child
    bread388 bread119 - bread-portion
    content337 content137 - content-portion
    tray106 tray354 tray269 - tray
    table8 table93 table344 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray106 kitchen)
     (at tray354 kitchen)
     (at tray269 kitchen)
     (at_kitchen_bread bread388)
     (at_kitchen_bread bread119)
     (at_kitchen_content content337)
     (at_kitchen_content content137)
     (not_allergic_gluten child120)
     (not_allergic_gluten child89)
     (waiting child120 table8)
     (waiting child89 table344)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child120)
     (served child89)
    )
  )
)
