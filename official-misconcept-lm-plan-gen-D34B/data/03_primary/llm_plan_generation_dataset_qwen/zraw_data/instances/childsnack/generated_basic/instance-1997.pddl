; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 868560

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child354 child214 - child
    bread343 bread345 - bread-portion
    content383 content60 - content-portion
    tray286 - tray
    table447 table288 table452 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray286 kitchen)
     (at_kitchen_bread bread343)
     (at_kitchen_bread bread345)
     (at_kitchen_content content383)
     (at_kitchen_content content60)
     (no_gluten_bread bread343)
     (no_gluten_content content383)
     (allergic_gluten child214)
     (not_allergic_gluten child354)
     (waiting child354 table288)
     (waiting child214 table452)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child354)
     (served child214)
    )
  )
)
