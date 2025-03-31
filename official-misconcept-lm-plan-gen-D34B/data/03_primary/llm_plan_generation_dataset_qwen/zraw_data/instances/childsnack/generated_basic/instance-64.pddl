; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 982666

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child5 child34 - child
    bread107 bread198 - bread-portion
    content4 content224 - content-portion
    tray319 tray244 - tray
    table279 table492 table220 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray319 kitchen)
     (at tray244 kitchen)
     (at_kitchen_bread bread107)
     (at_kitchen_bread bread198)
     (at_kitchen_content content4)
     (at_kitchen_content content224)
     (not_allergic_gluten child34)
     (not_allergic_gluten child5)
     (waiting child5 table492)
     (waiting child34 table279)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child5)
     (served child34)
    )
  )
)
