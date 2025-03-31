; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 64300

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child112 child420 - child
    bread402 bread279 - bread-portion
    content359 content414 - content-portion
    tray420 - tray
    table161 table356 table246 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray420 kitchen)
     (at_kitchen_bread bread402)
     (at_kitchen_bread bread279)
     (at_kitchen_content content359)
     (at_kitchen_content content414)
     (no_gluten_bread bread402)
     (no_gluten_content content359)
     (allergic_gluten child112)
     (not_allergic_gluten child420)
     (waiting child112 table356)
     (waiting child420 table161)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child112)
     (served child420)
    )
  )
)
