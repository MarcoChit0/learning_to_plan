; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 502039

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child219 child193 - child
    bread237 bread281 - bread-portion
    content352 content290 - content-portion
    tray276 - tray
    table162 table153 table197 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray276 kitchen)
     (at_kitchen_bread bread237)
     (at_kitchen_bread bread281)
     (at_kitchen_content content352)
     (at_kitchen_content content290)
     (not_allergic_gluten child219)
     (not_allergic_gluten child193)
     (waiting child219 table197)
     (waiting child193 table162)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child219)
     (served child193)
    )
  )
)
