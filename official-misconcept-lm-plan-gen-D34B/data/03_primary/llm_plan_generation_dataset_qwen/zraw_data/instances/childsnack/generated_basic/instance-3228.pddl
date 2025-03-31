; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 620854

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child188 child202 - child
    bread489 bread380 - bread-portion
    content64 content402 - content-portion
    tray490 tray465 tray414 tray92 - tray
    table480 table137 table193 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray490 kitchen)
     (at tray465 kitchen)
     (at tray414 kitchen)
     (at tray92 kitchen)
     (at_kitchen_bread bread489)
     (at_kitchen_bread bread380)
     (at_kitchen_content content64)
     (at_kitchen_content content402)
     (no_gluten_bread bread380)
     (no_gluten_content content402)
     (allergic_gluten child202)
     (not_allergic_gluten child188)
     (waiting child188 table480)
     (waiting child202 table193)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child188)
     (served child202)
    )
  )
)
