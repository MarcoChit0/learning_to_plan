; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 282356

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child169 child2 - child
    bread19 bread444 - bread-portion
    content203 content481 - content-portion
    tray498 - tray
    table161 table241 table126 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray498 kitchen)
     (at_kitchen_bread bread19)
     (at_kitchen_bread bread444)
     (at_kitchen_content content203)
     (at_kitchen_content content481)
     (no_gluten_bread bread19)
     (no_gluten_content content481)
     (allergic_gluten child2)
     (not_allergic_gluten child169)
     (waiting child169 table241)
     (waiting child2 table241)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child169)
     (served child2)
    )
  )
)
