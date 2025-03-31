; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 152244

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child488 child291 child347 - child
    bread250 bread121 bread88 - bread-portion
    content216 content463 content165 - content-portion
    tray222 - tray
    table366 table231 table34 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray222 kitchen)
     (at_kitchen_bread bread250)
     (at_kitchen_bread bread121)
     (at_kitchen_bread bread88)
     (at_kitchen_content content216)
     (at_kitchen_content content463)
     (at_kitchen_content content165)
     (not_allergic_gluten child291)
     (not_allergic_gluten child488)
     (not_allergic_gluten child347)
     (waiting child488 table366)
     (waiting child291 table366)
     (waiting child347 table366)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child488)
     (served child291)
     (served child347)
    )
  )
)
