; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 609828

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child137 child335 child345 - child
    bread54 bread460 bread128 - bread-portion
    content182 content425 content158 - content-portion
    tray115 - tray
    table90 table408 table230 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray115 kitchen)
     (at_kitchen_bread bread54)
     (at_kitchen_bread bread460)
     (at_kitchen_bread bread128)
     (at_kitchen_content content182)
     (at_kitchen_content content425)
     (at_kitchen_content content158)
     (not_allergic_gluten child335)
     (not_allergic_gluten child345)
     (not_allergic_gluten child137)
     (waiting child137 table408)
     (waiting child335 table90)
     (waiting child345 table230)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child137)
     (served child335)
     (served child345)
    )
  )
)
