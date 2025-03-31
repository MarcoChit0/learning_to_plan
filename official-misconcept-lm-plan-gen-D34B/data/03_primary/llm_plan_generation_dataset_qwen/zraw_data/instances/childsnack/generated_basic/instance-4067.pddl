; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 516309

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child345 - child
    bread110 - bread-portion
    content383 - content-portion
    tray434 tray351 tray289 - tray
    table236 table341 table300 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray434 kitchen)
     (at tray351 kitchen)
     (at tray289 kitchen)
     (at_kitchen_bread bread110)
     (at_kitchen_content content383)
     (not_allergic_gluten child345)
     (waiting child345 table236)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child345)
    )
  )
)
