; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 177212

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child25 child442 - child
    bread294 bread419 - bread-portion
    content480 content264 - content-portion
    tray99 tray34 tray318 - tray
    table452 table252 table96 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray99 kitchen)
     (at tray34 kitchen)
     (at tray318 kitchen)
     (at_kitchen_bread bread294)
     (at_kitchen_bread bread419)
     (at_kitchen_content content480)
     (at_kitchen_content content264)
     (not_allergic_gluten child442)
     (not_allergic_gluten child25)
     (waiting child25 table96)
     (waiting child442 table452)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child25)
     (served child442)
    )
  )
)
