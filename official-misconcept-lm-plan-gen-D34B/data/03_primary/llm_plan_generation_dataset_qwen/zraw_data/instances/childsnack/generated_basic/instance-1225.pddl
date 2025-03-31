; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 481727

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child93 child118 child391 - child
    bread73 bread483 bread493 - bread-portion
    content442 content111 content142 - content-portion
    tray404 tray276 tray149 - tray
    table428 table241 table437 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray404 kitchen)
     (at tray276 kitchen)
     (at tray149 kitchen)
     (at_kitchen_bread bread73)
     (at_kitchen_bread bread483)
     (at_kitchen_bread bread493)
     (at_kitchen_content content442)
     (at_kitchen_content content111)
     (at_kitchen_content content142)
     (not_allergic_gluten child391)
     (not_allergic_gluten child118)
     (not_allergic_gluten child93)
     (waiting child93 table241)
     (waiting child118 table428)
     (waiting child391 table428)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child93)
     (served child118)
     (served child391)
    )
  )
)
