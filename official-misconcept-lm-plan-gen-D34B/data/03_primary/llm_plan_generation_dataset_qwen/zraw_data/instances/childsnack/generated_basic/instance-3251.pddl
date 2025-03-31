; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 938729

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child481 child225 - child
    bread148 bread369 - bread-portion
    content237 content269 - content-portion
    tray399 tray49 tray187 tray325 - tray
    table329 table78 table418 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray399 kitchen)
     (at tray49 kitchen)
     (at tray187 kitchen)
     (at tray325 kitchen)
     (at_kitchen_bread bread148)
     (at_kitchen_bread bread369)
     (at_kitchen_content content237)
     (at_kitchen_content content269)
     (no_gluten_bread bread148)
     (no_gluten_content content237)
     (allergic_gluten child481)
     (not_allergic_gluten child225)
     (waiting child481 table78)
     (waiting child225 table418)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child481)
     (served child225)
    )
  )
)
