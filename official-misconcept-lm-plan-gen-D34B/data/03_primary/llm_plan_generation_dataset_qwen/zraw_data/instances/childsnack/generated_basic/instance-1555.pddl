; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 849309

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child179 child28 - child
    bread28 bread349 - bread-portion
    content206 content213 - content-portion
    tray181 tray451 tray112 tray226 - tray
    table353 table203 table345 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray181 kitchen)
     (at tray451 kitchen)
     (at tray112 kitchen)
     (at tray226 kitchen)
     (at_kitchen_bread bread28)
     (at_kitchen_bread bread349)
     (at_kitchen_content content206)
     (at_kitchen_content content213)
     (not_allergic_gluten child28)
     (not_allergic_gluten child179)
     (waiting child179 table345)
     (waiting child28 table345)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child179)
     (served child28)
    )
  )
)
