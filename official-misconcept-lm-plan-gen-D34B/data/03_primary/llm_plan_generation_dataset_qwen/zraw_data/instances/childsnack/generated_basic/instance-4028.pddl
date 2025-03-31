; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 967668

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child394 child5 child284 - child
    bread364 bread354 bread362 - bread-portion
    content265 content314 content437 - content-portion
    tray390 - tray
    table177 table305 table255 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray390 kitchen)
     (at_kitchen_bread bread364)
     (at_kitchen_bread bread354)
     (at_kitchen_bread bread362)
     (at_kitchen_content content265)
     (at_kitchen_content content314)
     (at_kitchen_content content437)
     (not_allergic_gluten child284)
     (not_allergic_gluten child394)
     (not_allergic_gluten child5)
     (waiting child394 table177)
     (waiting child5 table305)
     (waiting child284 table177)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child394)
     (served child5)
     (served child284)
    )
  )
)
