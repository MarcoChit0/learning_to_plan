; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 764750

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child417 child31 child167 - child
    bread306 bread241 bread228 - bread-portion
    content385 content340 content370 - content-portion
    tray213 tray404 tray381 - tray
    table387 table197 table272 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray213 kitchen)
     (at tray404 kitchen)
     (at tray381 kitchen)
     (at_kitchen_bread bread306)
     (at_kitchen_bread bread241)
     (at_kitchen_bread bread228)
     (at_kitchen_content content385)
     (at_kitchen_content content340)
     (at_kitchen_content content370)
     (not_allergic_gluten child417)
     (not_allergic_gluten child167)
     (not_allergic_gluten child31)
     (waiting child417 table197)
     (waiting child31 table272)
     (waiting child167 table387)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child417)
     (served child31)
     (served child167)
    )
  )
)
