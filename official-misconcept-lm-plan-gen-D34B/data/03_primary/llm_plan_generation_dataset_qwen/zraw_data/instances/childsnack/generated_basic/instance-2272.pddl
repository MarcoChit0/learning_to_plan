; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 321710

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child189 child99 - child
    bread351 bread75 - bread-portion
    content43 content438 - content-portion
    tray457 - tray
    table376 table408 table275 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray457 kitchen)
     (at_kitchen_bread bread351)
     (at_kitchen_bread bread75)
     (at_kitchen_content content43)
     (at_kitchen_content content438)
     (no_gluten_bread bread75)
     (no_gluten_content content438)
     (allergic_gluten child99)
     (not_allergic_gluten child189)
     (waiting child189 table408)
     (waiting child99 table408)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child189)
     (served child99)
    )
  )
)
