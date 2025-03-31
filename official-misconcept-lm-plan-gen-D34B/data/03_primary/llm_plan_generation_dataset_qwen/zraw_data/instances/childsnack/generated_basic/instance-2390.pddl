; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 753557

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child279 child294 - child
    bread15 bread178 - bread-portion
    content296 content164 - content-portion
    tray404 - tray
    table2 table487 table245 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray404 kitchen)
     (at_kitchen_bread bread15)
     (at_kitchen_bread bread178)
     (at_kitchen_content content296)
     (at_kitchen_content content164)
     (not_allergic_gluten child294)
     (not_allergic_gluten child279)
     (waiting child279 table2)
     (waiting child294 table487)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child279)
     (served child294)
    )
  )
)
