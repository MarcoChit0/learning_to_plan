; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 33814

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child253 child370 - child
    bread311 bread412 - bread-portion
    content258 content235 - content-portion
    tray82 tray344 - tray
    table298 table287 table176 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray82 kitchen)
     (at tray344 kitchen)
     (at_kitchen_bread bread311)
     (at_kitchen_bread bread412)
     (at_kitchen_content content258)
     (at_kitchen_content content235)
     (not_allergic_gluten child253)
     (not_allergic_gluten child370)
     (waiting child253 table298)
     (waiting child370 table176)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child253)
     (served child370)
    )
  )
)
