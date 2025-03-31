; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 494364

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child59 child467 - child
    bread295 bread276 - bread-portion
    content437 content408 - content-portion
    tray296 tray427 - tray
    table310 table335 table253 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray296 kitchen)
     (at tray427 kitchen)
     (at_kitchen_bread bread295)
     (at_kitchen_bread bread276)
     (at_kitchen_content content437)
     (at_kitchen_content content408)
     (not_allergic_gluten child467)
     (not_allergic_gluten child59)
     (waiting child59 table253)
     (waiting child467 table253)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child59)
     (served child467)
    )
  )
)
