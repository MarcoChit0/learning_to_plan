; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 531255

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child465 child215 child287 - child
    bread441 bread150 bread278 - bread-portion
    content309 content451 content243 - content-portion
    tray485 - tray
    table75 table473 table153 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray485 kitchen)
     (at_kitchen_bread bread441)
     (at_kitchen_bread bread150)
     (at_kitchen_bread bread278)
     (at_kitchen_content content309)
     (at_kitchen_content content451)
     (at_kitchen_content content243)
     (not_allergic_gluten child215)
     (not_allergic_gluten child465)
     (not_allergic_gluten child287)
     (waiting child465 table153)
     (waiting child215 table75)
     (waiting child287 table153)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child465)
     (served child215)
     (served child287)
    )
  )
)
