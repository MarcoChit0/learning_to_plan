; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 698036

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child350 child349 - child
    bread228 bread342 - bread-portion
    content275 content210 - content-portion
    tray415 tray145 tray430 - tray
    table490 table393 table340 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray415 kitchen)
     (at tray145 kitchen)
     (at tray430 kitchen)
     (at_kitchen_bread bread228)
     (at_kitchen_bread bread342)
     (at_kitchen_content content275)
     (at_kitchen_content content210)
     (no_gluten_bread bread342)
     (no_gluten_content content275)
     (allergic_gluten child349)
     (not_allergic_gluten child350)
     (waiting child350 table393)
     (waiting child349 table490)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child350)
     (served child349)
    )
  )
)
