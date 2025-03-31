; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 395554

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child301 child110 child12 - child
    bread458 bread154 bread182 - bread-portion
    content11 content122 content270 - content-portion
    tray234 - tray
    table289 table237 table122 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray234 kitchen)
     (at_kitchen_bread bread458)
     (at_kitchen_bread bread154)
     (at_kitchen_bread bread182)
     (at_kitchen_content content11)
     (at_kitchen_content content122)
     (at_kitchen_content content270)
     (not_allergic_gluten child12)
     (not_allergic_gluten child301)
     (not_allergic_gluten child110)
     (waiting child301 table289)
     (waiting child110 table289)
     (waiting child12 table237)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child301)
     (served child110)
     (served child12)
    )
  )
)
