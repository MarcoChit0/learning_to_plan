; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 607657

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child11 child187 - child
    bread54 bread465 - bread-portion
    content111 content369 - content-portion
    tray296 - tray
    table142 table444 table413 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray296 kitchen)
     (at_kitchen_bread bread54)
     (at_kitchen_bread bread465)
     (at_kitchen_content content111)
     (at_kitchen_content content369)
     (no_gluten_bread bread465)
     (no_gluten_content content111)
     (allergic_gluten child187)
     (not_allergic_gluten child11)
     (waiting child11 table142)
     (waiting child187 table444)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child11)
     (served child187)
    )
  )
)
