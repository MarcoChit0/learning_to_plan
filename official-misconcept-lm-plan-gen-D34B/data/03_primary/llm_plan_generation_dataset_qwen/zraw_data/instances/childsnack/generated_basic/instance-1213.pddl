; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 660264

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child284 child171 child188 - child
    bread87 bread385 bread243 - bread-portion
    content139 content369 content256 - content-portion
    tray260 tray98 tray178 - tray
    table9 table333 table270 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray260 kitchen)
     (at tray98 kitchen)
     (at tray178 kitchen)
     (at_kitchen_bread bread87)
     (at_kitchen_bread bread385)
     (at_kitchen_bread bread243)
     (at_kitchen_content content139)
     (at_kitchen_content content369)
     (at_kitchen_content content256)
     (not_allergic_gluten child284)
     (not_allergic_gluten child171)
     (not_allergic_gluten child188)
     (waiting child284 table333)
     (waiting child171 table333)
     (waiting child188 table9)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child284)
     (served child171)
     (served child188)
    )
  )
)
