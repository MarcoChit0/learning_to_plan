; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 80602

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child12 child173 child27 - child
    bread82 bread148 bread110 - bread-portion
    content465 content379 content317 - content-portion
    tray204 - tray
    table103 table3 table297 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray204 kitchen)
     (at_kitchen_bread bread82)
     (at_kitchen_bread bread148)
     (at_kitchen_bread bread110)
     (at_kitchen_content content465)
     (at_kitchen_content content379)
     (at_kitchen_content content317)
     (not_allergic_gluten child12)
     (not_allergic_gluten child27)
     (not_allergic_gluten child173)
     (waiting child12 table3)
     (waiting child173 table103)
     (waiting child27 table3)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child12)
     (served child173)
     (served child27)
    )
  )
)
