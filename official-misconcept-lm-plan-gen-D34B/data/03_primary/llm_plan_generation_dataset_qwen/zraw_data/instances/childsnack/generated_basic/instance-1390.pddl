; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 398948

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child458 child335 - child
    bread340 bread287 - bread-portion
    content471 content55 - content-portion
    tray207 - tray
    table5 table482 table459 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray207 kitchen)
     (at_kitchen_bread bread340)
     (at_kitchen_bread bread287)
     (at_kitchen_content content471)
     (at_kitchen_content content55)
     (no_gluten_bread bread287)
     (no_gluten_content content55)
     (allergic_gluten child458)
     (not_allergic_gluten child335)
     (waiting child458 table5)
     (waiting child335 table482)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child458)
     (served child335)
    )
  )
)
