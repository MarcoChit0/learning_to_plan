; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 361893

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child347 child124 - child
    bread360 bread138 - bread-portion
    content478 content165 - content-portion
    tray374 - tray
    table303 table470 table329 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray374 kitchen)
     (at_kitchen_bread bread360)
     (at_kitchen_bread bread138)
     (at_kitchen_content content478)
     (at_kitchen_content content165)
     (no_gluten_bread bread360)
     (no_gluten_content content478)
     (allergic_gluten child347)
     (not_allergic_gluten child124)
     (waiting child347 table329)
     (waiting child124 table303)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child347)
     (served child124)
    )
  )
)
