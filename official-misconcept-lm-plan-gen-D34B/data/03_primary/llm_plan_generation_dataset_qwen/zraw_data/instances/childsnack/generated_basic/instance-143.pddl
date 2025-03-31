; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 8466

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child55 child292 - child
    bread404 bread159 - bread-portion
    content261 content481 - content-portion
    tray276 tray404 tray243 tray462 - tray
    table97 table251 table402 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray276 kitchen)
     (at tray404 kitchen)
     (at tray243 kitchen)
     (at tray462 kitchen)
     (at_kitchen_bread bread404)
     (at_kitchen_bread bread159)
     (at_kitchen_content content261)
     (at_kitchen_content content481)
     (not_allergic_gluten child55)
     (not_allergic_gluten child292)
     (waiting child55 table251)
     (waiting child292 table251)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child55)
     (served child292)
    )
  )
)
