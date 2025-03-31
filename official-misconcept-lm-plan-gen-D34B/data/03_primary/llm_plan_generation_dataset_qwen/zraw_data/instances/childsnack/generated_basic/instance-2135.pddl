; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 438183

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child84 child298 - child
    bread261 bread265 - bread-portion
    content342 content8 - content-portion
    tray472 tray222 tray22 - tray
    table299 table490 table258 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray472 kitchen)
     (at tray222 kitchen)
     (at tray22 kitchen)
     (at_kitchen_bread bread261)
     (at_kitchen_bread bread265)
     (at_kitchen_content content342)
     (at_kitchen_content content8)
     (no_gluten_bread bread265)
     (no_gluten_content content8)
     (allergic_gluten child298)
     (not_allergic_gluten child84)
     (waiting child84 table299)
     (waiting child298 table490)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child84)
     (served child298)
    )
  )
)
