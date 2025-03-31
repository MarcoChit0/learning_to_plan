; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 699865

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child270 child364 - child
    bread307 bread467 - bread-portion
    content129 content163 - content-portion
    tray319 tray163 tray231 tray232 - tray
    table476 table280 table464 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray319 kitchen)
     (at tray163 kitchen)
     (at tray231 kitchen)
     (at tray232 kitchen)
     (at_kitchen_bread bread307)
     (at_kitchen_bread bread467)
     (at_kitchen_content content129)
     (at_kitchen_content content163)
     (no_gluten_bread bread467)
     (no_gluten_content content129)
     (allergic_gluten child364)
     (not_allergic_gluten child270)
     (waiting child270 table476)
     (waiting child364 table464)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child270)
     (served child364)
    )
  )
)
