; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 406350

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child477 child89 child49 - child
    bread300 bread297 bread488 - bread-portion
    content309 content186 content127 - content-portion
    tray444 - tray
    table396 table20 table477 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray444 kitchen)
     (at_kitchen_bread bread300)
     (at_kitchen_bread bread297)
     (at_kitchen_bread bread488)
     (at_kitchen_content content309)
     (at_kitchen_content content186)
     (at_kitchen_content content127)
     (not_allergic_gluten child89)
     (not_allergic_gluten child49)
     (not_allergic_gluten child477)
     (waiting child477 table20)
     (waiting child89 table396)
     (waiting child49 table396)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child477)
     (served child89)
     (served child49)
    )
  )
)
