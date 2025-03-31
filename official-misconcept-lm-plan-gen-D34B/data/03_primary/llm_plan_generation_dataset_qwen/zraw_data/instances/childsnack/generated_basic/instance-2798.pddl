; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 527371

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child273 child458 - child
    bread457 bread285 - bread-portion
    content150 content313 - content-portion
    tray483 tray475 - tray
    table301 table342 table318 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray483 kitchen)
     (at tray475 kitchen)
     (at_kitchen_bread bread457)
     (at_kitchen_bread bread285)
     (at_kitchen_content content150)
     (at_kitchen_content content313)
     (not_allergic_gluten child273)
     (not_allergic_gluten child458)
     (waiting child273 table342)
     (waiting child458 table301)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child273)
     (served child458)
    )
  )
)
