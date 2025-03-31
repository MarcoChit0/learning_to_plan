; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 919269

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child89 child378 - child
    bread475 bread257 - bread-portion
    content362 content228 - content-portion
    tray457 tray153 tray475 tray175 - tray
    table73 table0 table290 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray457 kitchen)
     (at tray153 kitchen)
     (at tray475 kitchen)
     (at tray175 kitchen)
     (at_kitchen_bread bread475)
     (at_kitchen_bread bread257)
     (at_kitchen_content content362)
     (at_kitchen_content content228)
     (not_allergic_gluten child89)
     (not_allergic_gluten child378)
     (waiting child89 table0)
     (waiting child378 table290)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child89)
     (served child378)
    )
  )
)
