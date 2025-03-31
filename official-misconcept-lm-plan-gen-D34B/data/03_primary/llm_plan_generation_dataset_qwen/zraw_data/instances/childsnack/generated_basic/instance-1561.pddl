; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 263520

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child377 child82 - child
    bread381 bread215 - bread-portion
    content189 content269 - content-portion
    tray458 tray379 tray273 tray212 - tray
    table304 table447 table347 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray458 kitchen)
     (at tray379 kitchen)
     (at tray273 kitchen)
     (at tray212 kitchen)
     (at_kitchen_bread bread381)
     (at_kitchen_bread bread215)
     (at_kitchen_content content189)
     (at_kitchen_content content269)
     (not_allergic_gluten child377)
     (not_allergic_gluten child82)
     (waiting child377 table304)
     (waiting child82 table304)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child377)
     (served child82)
    )
  )
)
