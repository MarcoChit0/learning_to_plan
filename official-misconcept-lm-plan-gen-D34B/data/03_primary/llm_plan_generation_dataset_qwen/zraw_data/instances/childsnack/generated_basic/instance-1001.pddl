; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 216137

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child251 child106 child348 - child
    bread367 bread51 bread391 - bread-portion
    content135 content388 content461 - content-portion
    tray168 - tray
    table95 table94 table115 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray168 kitchen)
     (at_kitchen_bread bread367)
     (at_kitchen_bread bread51)
     (at_kitchen_bread bread391)
     (at_kitchen_content content135)
     (at_kitchen_content content388)
     (at_kitchen_content content461)
     (not_allergic_gluten child348)
     (not_allergic_gluten child251)
     (not_allergic_gluten child106)
     (waiting child251 table95)
     (waiting child106 table115)
     (waiting child348 table115)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child251)
     (served child106)
     (served child348)
    )
  )
)
