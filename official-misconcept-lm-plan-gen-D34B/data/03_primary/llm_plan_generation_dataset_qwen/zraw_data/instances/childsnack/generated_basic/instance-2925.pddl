; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 306449

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child112 child394 - child
    bread58 bread484 - bread-portion
    content269 content208 - content-portion
    tray336 tray367 tray398 - tray
    table269 table207 table484 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray336 kitchen)
     (at tray367 kitchen)
     (at tray398 kitchen)
     (at_kitchen_bread bread58)
     (at_kitchen_bread bread484)
     (at_kitchen_content content269)
     (at_kitchen_content content208)
     (no_gluten_bread bread58)
     (no_gluten_content content208)
     (allergic_gluten child394)
     (not_allergic_gluten child112)
     (waiting child112 table269)
     (waiting child394 table484)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child112)
     (served child394)
    )
  )
)
