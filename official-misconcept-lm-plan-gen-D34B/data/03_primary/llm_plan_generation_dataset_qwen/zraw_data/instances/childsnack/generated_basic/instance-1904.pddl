; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 598460

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child440 child310 - child
    bread414 bread121 - bread-portion
    content373 content22 - content-portion
    tray161 tray449 - tray
    table451 table79 table382 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray161 kitchen)
     (at tray449 kitchen)
     (at_kitchen_bread bread414)
     (at_kitchen_bread bread121)
     (at_kitchen_content content373)
     (at_kitchen_content content22)
     (no_gluten_bread bread121)
     (no_gluten_content content22)
     (allergic_gluten child310)
     (not_allergic_gluten child440)
     (waiting child440 table451)
     (waiting child310 table382)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child440)
     (served child310)
    )
  )
)
