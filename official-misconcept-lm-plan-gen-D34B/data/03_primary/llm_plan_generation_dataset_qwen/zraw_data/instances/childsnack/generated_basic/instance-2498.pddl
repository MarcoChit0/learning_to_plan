; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 335563

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child127 child370 - child
    bread301 bread4 - bread-portion
    content268 content347 - content-portion
    tray295 tray188 - tray
    table76 table385 table420 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray295 kitchen)
     (at tray188 kitchen)
     (at_kitchen_bread bread301)
     (at_kitchen_bread bread4)
     (at_kitchen_content content268)
     (at_kitchen_content content347)
     (not_allergic_gluten child127)
     (not_allergic_gluten child370)
     (waiting child127 table385)
     (waiting child370 table385)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child127)
     (served child370)
    )
  )
)
