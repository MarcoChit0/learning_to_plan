; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 542932

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 - child
    bread345 - bread-portion
    content355 - content-portion
    tray486 tray370 tray497 - tray
    table405 table222 table364 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray486 kitchen)
     (at tray370 kitchen)
     (at tray497 kitchen)
     (at_kitchen_bread bread345)
     (at_kitchen_content content355)
     (not_allergic_gluten child431)
     (waiting child431 table222)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child431)
    )
  )
)
