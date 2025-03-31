; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 537473

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child442 child32 - child
    bread247 bread345 - bread-portion
    content481 content16 - content-portion
    tray135 tray7 tray71 - tray
    table412 table318 table267 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray135 kitchen)
     (at tray7 kitchen)
     (at tray71 kitchen)
     (at_kitchen_bread bread247)
     (at_kitchen_bread bread345)
     (at_kitchen_content content481)
     (at_kitchen_content content16)
     (no_gluten_bread bread345)
     (no_gluten_content content481)
     (allergic_gluten child32)
     (not_allergic_gluten child442)
     (waiting child442 table267)
     (waiting child32 table267)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child442)
     (served child32)
    )
  )
)
