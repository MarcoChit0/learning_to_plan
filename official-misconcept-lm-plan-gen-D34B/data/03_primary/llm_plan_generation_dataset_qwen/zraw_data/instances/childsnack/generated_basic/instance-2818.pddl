; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 359047

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child274 child66 - child
    bread480 bread332 - bread-portion
    content491 content2 - content-portion
    tray91 tray161 - tray
    table301 table322 table466 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray91 kitchen)
     (at tray161 kitchen)
     (at_kitchen_bread bread480)
     (at_kitchen_bread bread332)
     (at_kitchen_content content491)
     (at_kitchen_content content2)
     (not_allergic_gluten child274)
     (not_allergic_gluten child66)
     (waiting child274 table301)
     (waiting child66 table322)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child274)
     (served child66)
    )
  )
)
