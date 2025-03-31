; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 951452

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child151 child175 - child
    bread476 bread88 - bread-portion
    content217 content158 - content-portion
    tray481 - tray
    table456 table128 table483 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray481 kitchen)
     (at_kitchen_bread bread476)
     (at_kitchen_bread bread88)
     (at_kitchen_content content217)
     (at_kitchen_content content158)
     (not_allergic_gluten child175)
     (not_allergic_gluten child151)
     (waiting child151 table456)
     (waiting child175 table128)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child151)
     (served child175)
    )
  )
)
