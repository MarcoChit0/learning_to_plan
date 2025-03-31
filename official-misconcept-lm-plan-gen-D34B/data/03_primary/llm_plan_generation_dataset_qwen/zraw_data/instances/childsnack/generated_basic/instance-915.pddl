; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 267774

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child218 child440 child314 - child
    bread119 bread426 bread326 - bread-portion
    content496 content449 content465 - content-portion
    tray220 - tray
    table379 table312 table248 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray220 kitchen)
     (at_kitchen_bread bread119)
     (at_kitchen_bread bread426)
     (at_kitchen_bread bread326)
     (at_kitchen_content content496)
     (at_kitchen_content content449)
     (at_kitchen_content content465)
     (not_allergic_gluten child314)
     (not_allergic_gluten child218)
     (not_allergic_gluten child440)
     (waiting child218 table248)
     (waiting child440 table248)
     (waiting child314 table248)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child218)
     (served child440)
     (served child314)
    )
  )
)
