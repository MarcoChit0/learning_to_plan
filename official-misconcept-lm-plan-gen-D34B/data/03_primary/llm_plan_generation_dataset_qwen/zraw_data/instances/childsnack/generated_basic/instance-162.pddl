; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 691401

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child350 child294 - child
    bread378 bread458 - bread-portion
    content228 content61 - content-portion
    tray8 tray35 tray110 tray310 - tray
    table248 table236 table115 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray8 kitchen)
     (at tray35 kitchen)
     (at tray110 kitchen)
     (at tray310 kitchen)
     (at_kitchen_bread bread378)
     (at_kitchen_bread bread458)
     (at_kitchen_content content228)
     (at_kitchen_content content61)
     (not_allergic_gluten child294)
     (not_allergic_gluten child350)
     (waiting child350 table236)
     (waiting child294 table248)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child350)
     (served child294)
    )
  )
)
