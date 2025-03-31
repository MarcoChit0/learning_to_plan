; child-snack task with 2 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 869313

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child323 child219 - child
    bread366 bread349 - bread-portion
    content103 content128 - content-portion
    tray288 - tray
    table181 table193 table187 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray288 kitchen)
     (at_kitchen_bread bread366)
     (at_kitchen_bread bread349)
     (at_kitchen_content content103)
     (at_kitchen_content content128)
     (no_gluten_bread bread366)
     (no_gluten_content content128)
     (allergic_gluten child323)
     (not_allergic_gluten child219)
     (waiting child323 table193)
     (waiting child219 table181)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child323)
     (served child219)
    )
  )
)
