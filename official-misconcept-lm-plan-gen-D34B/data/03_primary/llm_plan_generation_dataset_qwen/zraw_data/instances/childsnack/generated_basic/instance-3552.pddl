; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 43536

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child405 child463 - child
    bread174 bread90 - bread-portion
    content337 content103 - content-portion
    tray451 tray323 - tray
    table471 table312 table272 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray451 kitchen)
     (at tray323 kitchen)
     (at_kitchen_bread bread174)
     (at_kitchen_bread bread90)
     (at_kitchen_content content337)
     (at_kitchen_content content103)
     (not_allergic_gluten child463)
     (not_allergic_gluten child405)
     (waiting child405 table471)
     (waiting child463 table272)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child405)
     (served child463)
    )
  )
)
