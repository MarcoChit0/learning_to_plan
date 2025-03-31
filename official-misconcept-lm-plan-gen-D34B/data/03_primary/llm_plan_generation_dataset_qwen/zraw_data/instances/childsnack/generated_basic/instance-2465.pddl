; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 738318

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child275 child307 - child
    bread426 bread324 - bread-portion
    content203 content372 - content-portion
    tray414 tray84 - tray
    table146 table417 table75 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray414 kitchen)
     (at tray84 kitchen)
     (at_kitchen_bread bread426)
     (at_kitchen_bread bread324)
     (at_kitchen_content content203)
     (at_kitchen_content content372)
     (not_allergic_gluten child307)
     (not_allergic_gluten child275)
     (waiting child275 table417)
     (waiting child307 table146)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child275)
     (served child307)
    )
  )
)
