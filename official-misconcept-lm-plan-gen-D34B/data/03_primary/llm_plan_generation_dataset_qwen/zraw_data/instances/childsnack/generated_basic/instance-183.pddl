; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 505447

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child109 child485 - child
    bread269 bread477 - bread-portion
    content417 content369 - content-portion
    tray144 tray361 tray99 - tray
    table444 table291 table462 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray144 kitchen)
     (at tray361 kitchen)
     (at tray99 kitchen)
     (at_kitchen_bread bread269)
     (at_kitchen_bread bread477)
     (at_kitchen_content content417)
     (at_kitchen_content content369)
     (not_allergic_gluten child109)
     (not_allergic_gluten child485)
     (waiting child109 table291)
     (waiting child485 table291)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child109)
     (served child485)
    )
  )
)
