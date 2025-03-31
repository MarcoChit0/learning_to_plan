; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 894428

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child121 child385 - child
    bread452 bread316 - bread-portion
    content470 content480 - content-portion
    tray327 tray431 - tray
    table183 table335 table416 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray327 kitchen)
     (at tray431 kitchen)
     (at_kitchen_bread bread452)
     (at_kitchen_bread bread316)
     (at_kitchen_content content470)
     (at_kitchen_content content480)
     (no_gluten_bread bread452)
     (no_gluten_content content470)
     (allergic_gluten child121)
     (not_allergic_gluten child385)
     (waiting child121 table335)
     (waiting child385 table416)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child121)
     (served child385)
    )
  )
)
