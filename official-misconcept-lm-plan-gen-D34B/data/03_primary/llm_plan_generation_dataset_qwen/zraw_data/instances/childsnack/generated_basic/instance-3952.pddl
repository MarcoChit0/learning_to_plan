; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 937882

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child404 child66 - child
    bread361 bread268 - bread-portion
    content477 content382 - content-portion
    tray456 - tray
    table369 table82 table301 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray456 kitchen)
     (at_kitchen_bread bread361)
     (at_kitchen_bread bread268)
     (at_kitchen_content content477)
     (at_kitchen_content content382)
     (no_gluten_bread bread361)
     (no_gluten_content content382)
     (allergic_gluten child66)
     (not_allergic_gluten child404)
     (waiting child404 table301)
     (waiting child66 table82)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child404)
     (served child66)
    )
  )
)
