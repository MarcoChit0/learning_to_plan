; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 969777

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child349 child57 child164 - child
    bread374 bread440 bread40 - bread-portion
    content156 content30 content235 - content-portion
    tray216 - tray
    table458 table272 table456 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray216 kitchen)
     (at_kitchen_bread bread374)
     (at_kitchen_bread bread440)
     (at_kitchen_bread bread40)
     (at_kitchen_content content156)
     (at_kitchen_content content30)
     (at_kitchen_content content235)
     (not_allergic_gluten child349)
     (not_allergic_gluten child164)
     (not_allergic_gluten child57)
     (waiting child349 table458)
     (waiting child57 table456)
     (waiting child164 table456)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child349)
     (served child57)
     (served child164)
    )
  )
)
