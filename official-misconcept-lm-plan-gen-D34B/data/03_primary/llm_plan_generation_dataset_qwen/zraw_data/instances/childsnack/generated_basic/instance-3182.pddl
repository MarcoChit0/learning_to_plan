; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 883933

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child62 child345 - child
    bread45 bread286 - bread-portion
    content496 content295 - content-portion
    tray203 tray430 - tray
    table398 table437 table94 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray203 kitchen)
     (at tray430 kitchen)
     (at_kitchen_bread bread45)
     (at_kitchen_bread bread286)
     (at_kitchen_content content496)
     (at_kitchen_content content295)
     (not_allergic_gluten child345)
     (not_allergic_gluten child62)
     (waiting child62 table94)
     (waiting child345 table398)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child62)
     (served child345)
    )
  )
)
