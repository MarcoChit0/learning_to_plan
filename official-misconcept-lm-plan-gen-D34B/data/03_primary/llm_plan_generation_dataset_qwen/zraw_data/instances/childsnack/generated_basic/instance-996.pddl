; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 979662

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child135 child22 child368 - child
    bread122 bread395 bread440 - bread-portion
    content379 content329 content60 - content-portion
    tray215 - tray
    table90 table152 table154 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray215 kitchen)
     (at_kitchen_bread bread122)
     (at_kitchen_bread bread395)
     (at_kitchen_bread bread440)
     (at_kitchen_content content379)
     (at_kitchen_content content329)
     (at_kitchen_content content60)
     (not_allergic_gluten child22)
     (not_allergic_gluten child368)
     (not_allergic_gluten child135)
     (waiting child135 table154)
     (waiting child22 table90)
     (waiting child368 table154)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child135)
     (served child22)
     (served child368)
    )
  )
)
