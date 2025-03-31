; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 852602

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child274 child456 child394 - child
    bread469 bread494 bread269 - bread-portion
    content161 content270 content119 - content-portion
    tray334 tray421 tray225 - tray
    table43 table58 table475 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray334 kitchen)
     (at tray421 kitchen)
     (at tray225 kitchen)
     (at_kitchen_bread bread469)
     (at_kitchen_bread bread494)
     (at_kitchen_bread bread269)
     (at_kitchen_content content161)
     (at_kitchen_content content270)
     (at_kitchen_content content119)
     (not_allergic_gluten child274)
     (not_allergic_gluten child394)
     (not_allergic_gluten child456)
     (waiting child274 table58)
     (waiting child456 table43)
     (waiting child394 table43)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child274)
     (served child456)
     (served child394)
    )
  )
)
