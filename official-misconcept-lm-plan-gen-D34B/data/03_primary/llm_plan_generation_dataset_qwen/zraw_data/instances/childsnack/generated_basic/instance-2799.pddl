; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 20740

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child368 child180 - child
    bread394 bread196 - bread-portion
    content445 content225 - content-portion
    tray88 tray179 - tray
    table59 table422 table69 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray88 kitchen)
     (at tray179 kitchen)
     (at_kitchen_bread bread394)
     (at_kitchen_bread bread196)
     (at_kitchen_content content445)
     (at_kitchen_content content225)
     (not_allergic_gluten child368)
     (not_allergic_gluten child180)
     (waiting child368 table59)
     (waiting child180 table59)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child368)
     (served child180)
    )
  )
)
