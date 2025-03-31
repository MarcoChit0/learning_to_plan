; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 846053

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child203 child390 - child
    bread456 bread381 - bread-portion
    content339 content59 - content-portion
    tray384 tray39 - tray
    table55 table312 table374 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray384 kitchen)
     (at tray39 kitchen)
     (at_kitchen_bread bread456)
     (at_kitchen_bread bread381)
     (at_kitchen_content content339)
     (at_kitchen_content content59)
     (not_allergic_gluten child203)
     (not_allergic_gluten child390)
     (waiting child203 table374)
     (waiting child390 table374)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child203)
     (served child390)
    )
  )
)
